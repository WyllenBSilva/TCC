#!/usr/bin/env python3
"""
etapa3_alvo.py
==============
Etapa 3 do pipeline: agrega a base limpa na série mensal de pacientes-dia, com
reconstrução das lacunas de longa permanência e auditoria do resultado.

Uso
---
    python pipeline/etapa3_alvo.py

Entrada (somente leitura):
    pipeline/saidas/etapa1_base_limpa.parquet

Saídas (em pipeline/saidas/):
    etapa3_serie_mensal.parquet    01/2015 a 08/2025 (128 meses); a Etapa 4 lê este
    etapa3_serie_completa.parquet  01/2015 a 12/2025 (132 meses); auditoria e censura
    etapa3_relatorio.txt           contagens, auditoria e conferência com a referência

Colunas das séries
------------------
    mes                             primeiro dia do mês
    pacientes_dia                   alvo principal: zeros = 0, com reconstrução
    pacientes_dia_sem_reconstrucao  zeros = 0, sem as lacunas (sensibilidade)
    pacientes_dia_zeros1            zeros = 1, com reconstrução (sensibilidade)
    pd_reconstruidos                pacientes-dia das lacunas
    pd_masculino                    parte de pacientes_dia com SEXO = 1
    pd_idade_valida                 parte de pacientes_dia com IDADE_ANOS não nula
    pd_60mais                       parte de pacientes_dia com IDADE_ANOS >= 60

Regras (ver CLAUDE.md)
----------------------
- IDENT != 5: os dias de [DT_INTER, DT_SAIDA) são distribuídos pelos meses que o
  intervalo cobre. O dia da internação conta, o da alta não.
- IDENT  = 5: DIAS_PERM é lançado inteiro no mês de DT_SAIDA. O DT_INTER dessas
  fatias nunca é usado na alocação; só se confere se ele muda ao longo da cadeia.
- Lacuna: mês sem fatia entre duas fatias IDENT = 5 consecutivas da mesma N_AIH. Vale
  o número de dias do mês, com sexo e idade da fatia anterior.
- Nada é deduplicado nem filtrado por idade. Os dias anteriores a 01/2015 ficam fora
  das séries, mas os registros são lidos, pois a parte deles na janela é ocupação real.

Auditoria
---------
1. Por registro: a soma das parcelas de cada IDENT != 5 tem de ser igual a DIAS_PERM.
2. Global: tudo o que foi alocado, em todos os meses, tem de somar DIAS_PERM da base.
3. Método independente: cada mês é recalculado pela sobreposição de intervalos, sem o
   vetor de cobertura usado na alocação, e tem de dar o mesmo valor.
4. Decomposição do diagnóstico e valores de referência do CLAUDE.md.
Se algum item divergir, o relatório é gravado, mas as séries NÃO.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ      = Path(__file__).resolve().parent.parent
SAIDA     = RAIZ / "pipeline" / "saidas"
ENTRADA   = SAIDA / "etapa1_base_limpa.parquet"
ARQ_SERIE = SAIDA / "etapa3_serie_mensal.parquet"
ARQ_COMPL = SAIDA / "etapa3_serie_completa.parquet"
ARQ_REL   = SAIDA / "etapa3_relatorio.txt"

TAM_LOTE = 250_000
COLUNAS  = ["N_AIH", "IDENT", "DT_INTER", "DT_SAIDA", "DIAS_PERM", "IDADE_ANOS", "SEXO"]

# Calendário interno: cobre todas as datas da base (DT_INTER mínimo: 2008-01-01)
MES_BASE = np.datetime64("2008-01")
N_MESES  = (2025 - 2008 + 1) * 12
# INICIO[k] = primeiro dia do mês k, em dias desde a época (tem N_MESES + 1 posições)
INICIO   = (MES_BASE + np.arange(N_MESES + 1)).astype("datetime64[D]").astype("int64")
DIAS_MES = np.diff(INICIO)


def indice_mes(texto: str) -> int:
    return int((np.datetime64(texto) - MES_BASE).astype("int64"))


I_JANELA    = indice_mes("2015-01")   # primeiro mês das séries
I_FIM_MODEL = indice_mes("2025-09")   # exclusivo: série de modelagem termina em 08/2025
I_FIM_COMPL = indice_mes("2026-01")   # exclusivo: série completa termina em 12/2025

# Valores de referência (CLAUDE.md, obtidos pelos diagnósticos em tests/saidas/)
REFERENCIA = {
    "registros": 5_758_981,
    "fatias IDENT = 5": 30_323,
    "soma de DIAS_PERM": 26_304_466,
    "soma de DIAS_PERM (IDENT = 5)": 858_794,
    "auditoria: IDENT != 5 com parcelas != DIAS_PERM": 0,
    "auditoria: alocado - DIAS_PERM da base": 0,
    "auditoria: meses divergentes no método independente": 0,
    # O CLAUDE.md divide o total em 25.248.838 + 858.794 + 31.452, mas 858.794 inclui
    # 22 fatias com saída em 2014 (662 dias), fora da janela. A soma é a mesma; aqui
    # esses 662 dias saem de IDENT = 5 e o item de borda fica com o valor medido.
    "fatias com saída antes de 01/2015 (dias)": 662,
    "decomposição: IDENT != 5 inteiro na janela até 12/2025": 25_248_838,
    "decomposição: IDENT = 5": 858_132,
    "decomposição: parcelas de borda": 32_114,
    "total 01/2015-12/2025, sem reconstrução": 26_139_084,
    "buracos entre fatias": 256,
    "buracos de um mês": 235,
    "lacunas": 298,
    "pacientes-dia das lacunas": 9_018,
    "meses na série completa": 132,
    "meses na série de modelagem": 128,
    "total do alvo (zeros = 0, com reconstrução)": 25_430_958,
    "total do alvo (zeros = 0, sem reconstrução)": 25_421_940,
    "total do alvo (zeros = 1, com reconstrução)": 26_038_944,
    "mês 2015-01": 177_864,
    "mês 2020-04": 137_807,
    "mês 2021-11": 194_535,
    "mês 2025-08": 243_258,
    "pacientes-dia com idade inválida": 54,
}


# ============================================================
# UTILITÁRIOS
# ============================================================

_linhas: list[str] = []


def log(texto="") -> None:
    """Imprime e acumula o texto para gravar no relatório."""
    print(texto)
    _linhas.append(str(texto))


def titulo(texto: str) -> None:
    log("\n" + "=" * 74)
    log(texto)
    log("=" * 74)


def dias_inteiros(s: pd.Series) -> np.ndarray:
    """Datas -> dias desde a época."""
    return s.to_numpy().astype("datetime64[D]").astype("int64")


def meses_inteiros(s: pd.Series) -> np.ndarray:
    """Datas -> índice no calendário interno."""
    return (s.to_numpy().astype("datetime64[M]") - MES_BASE).astype("int64")


def rotulo_mes(indice: int) -> str:
    return str(MES_BASE + np.timedelta64(indice, "M"))


# ============================================================
# ALOCAÇÃO DOS PACIENTES-DIA
# ============================================================

class Alocacao:
    """Pacientes-dia por mês. Os meses inteiros de cada intervalo vão para um vetor de
    cobertura, que conta quantos registros ocupam o mês inteiro."""

    def __init__(self) -> None:
        self.pontual = np.zeros(N_MESES, dtype=np.int64)
        self.cobertura = np.zeros(N_MESES + 1, dtype=np.int64)

    def intervalos(self, mi, mf, primeira, ultima) -> None:
        np.add.at(self.pontual, mi, primeira)
        np.add.at(self.pontual, mf, ultima)
        longos = mf > mi + 1
        np.add.at(self.cobertura, mi[longos] + 1, 1)
        np.add.at(self.cobertura, mf[longos], -1)

    def pontos(self, mes, valor) -> None:
        np.add.at(self.pontual, mes, valor)

    def serie(self) -> np.ndarray:
        return self.pontual + np.cumsum(self.cobertura)[:N_MESES] * DIAS_MES


class Acumulador:
    def __init__(self, aih_fatias: np.ndarray) -> None:
        self.cont = Counter()
        self.aih_fatias = aih_fatias
        self.grupos = {nome: Alocacao() for nome in ["total", "masculino", "idade_valida", "60mais"]}
        self.zeros = np.zeros(N_MESES, dtype=np.int64)
        self.independente = np.zeros(N_MESES, dtype=np.int64)
        self.aberturas: list[pd.DataFrame] = []

    def adicionar(self, d: pd.DataFrame) -> None:
        c = self.cont
        c["registros"] += len(d)
        is5 = (d["IDENT"] == 5).to_numpy()
        dias = d["DIAS_PERM"].to_numpy().astype("int64")
        ini, fim = dias_inteiros(d["DT_INTER"]), dias_inteiros(d["DT_SAIDA"])
        mi, mf = meses_inteiros(d["DT_INTER"]), meses_inteiros(d["DT_SAIDA"])
        if mf.min() < 0 or mf.max() >= N_MESES or mi[~is5].min() < 0:
            sys.exit("Data fora do calendário interno; ajustar MES_BASE ou N_MESES.")

        c["soma de DIAS_PERM"] += int(dias.sum())
        c["fatias IDENT = 5"] += int(is5.sum())
        c["soma de DIAS_PERM (IDENT = 5)"] += int(dias[is5].sum())

        # IDENT != 5: primeira parcela, última parcela e meses inteiros entre elas
        co = ~is5
        mi_c, mf_c, ini_c, fim_c, dias_c = mi[co], mf[co], ini[co], fim[co], dias[co]
        mesmo = mi_c == mf_c
        primeira = np.where(mesmo, fim_c - ini_c, INICIO[mi_c + 1] - ini_c)
        ultima = np.where(mesmo, 0, fim_c - INICIO[mf_c])
        meio = np.where(mf_c > mi_c, INICIO[mf_c] - INICIO[np.minimum(mi_c + 1, mf_c)], 0)
        c["auditoria: IDENT != 5 com parcelas != DIAS_PERM"] += int(
            (primeira + ultima + meio != dias_c).sum())

        # decomposição do diagnóstico, na janela 01/2015 a 12/2025 e sem reconstrução
        a, b = INICIO[I_JANELA], INICIO[I_FIM_COMPL]
        inteiro = (ini_c >= a) & (fim_c <= b)
        c["decomposição: IDENT != 5 inteiro na janela até 12/2025"] += int(dias_c[inteiro].sum())
        borda = np.minimum(fim_c[~inteiro], b) - np.maximum(ini_c[~inteiro], a)
        c["decomposição: parcelas de borda"] += int(np.clip(borda, 0, None).sum())
        c["decomposição: IDENT = 5"] += int(dias[is5 & (fim >= a) & (fim < b)].sum())
        c["fatias com saída antes de 01/2015 (dias)"] += int(dias[is5 & (fim < a)].sum())

        # grupos usados na proporção por sexo e por idade (Etapa 4)
        idade = d["IDADE_ANOS"].to_numpy(dtype=float)
        mascaras = {
            "total": np.ones(len(d), dtype=bool),
            "masculino": (d["SEXO"] == "1").to_numpy(),
            "idade_valida": ~np.isnan(idade),
            "60mais": idade >= 60,
        }
        for nome, m in mascaras.items():
            g = self.grupos[nome]
            mc = m[co]
            g.intervalos(mi_c[mc], mf_c[mc], primeira[mc], ultima[mc])
            g.pontos(mf[is5 & m], dias[is5 & m])

        # zeros = 1: cada saída no mesmo dia conta como um dia (só ocorre em IDENT != 5)
        np.add.at(self.zeros, mi_c[dias_c == 0], 1)

        # recálculo por sobreposição de intervalos, sem o vetor de cobertura
        positivos = dias_c > 0
        i_p, f_p = ini_c[positivos], fim_c[positivos]
        fim5, dias5 = fim[is5], dias[is5]
        for k in range(I_JANELA, I_FIM_COMPL):
            a, b = INICIO[k], INICIO[k + 1]
            sobrepostos = np.minimum(f_p, b) - np.maximum(i_p, a)
            self.independente[k] += int(np.clip(sobrepostos, 0, None).sum())
            self.independente[k] += int(dias5[(fim5 >= a) & (fim5 < b)].sum())

        # abertura das cadeias, usada só para medir o mês ausente antes da primeira fatia
        aih = d["N_AIH"].to_numpy()
        abre = co & np.isin(aih, self.aih_fatias)
        self.aberturas.append(pd.DataFrame({"N_AIH": aih[abre], "mes": mf[abre]}))


def ler_fatias() -> pd.DataFrame:
    """Fatias IDENT = 5, ordenadas por N_AIH e mês de saída."""
    f = pq.read_table(
        ENTRADA,
        columns=["N_AIH", "DT_INTER", "DT_SAIDA", "IDADE_ANOS", "SEXO", "NASC", "MUNIC_RES",
                 "DIAG_PRINC"],
        filters=[("IDENT", "=", 5)],
    ).to_pandas()
    for nome in ["SEXO", "MUNIC_RES", "DIAG_PRINC"]:
        f[nome] = f[nome].astype(str)
    f["mes"] = meses_inteiros(f["DT_SAIDA"])
    return f.sort_values(["N_AIH", "mes"], ignore_index=True)


def processar(aih_fatias: np.ndarray) -> Acumulador:
    acc = Acumulador(aih_fatias)
    for lote in pq.ParquetFile(ENTRADA).iter_batches(batch_size=TAM_LOTE, columns=COLUNAS):
        acc.adicionar(lote.to_pandas())
        print(f"  ... {acc.cont['registros']:,} registros alocados", end="\r", flush=True)
    print()
    return acc


# ============================================================
# LACUNAS
# ============================================================

def achar_lacunas(f: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por mês ausente entre duas fatias consecutivas da mesma N_AIH."""
    mesma = f["N_AIH"] == f["N_AIH"].shift()
    salto = f["mes"] - f["mes"].shift()
    depois = f[mesma & (salto > 1)].index
    linhas = []
    for pos in depois:
        ant = f.loc[pos - 1]
        for mes in range(int(ant["mes"]) + 1, int(f.at[pos, "mes"])):
            linhas.append({
                "N_AIH": ant["N_AIH"], "mes": mes, "buraco": pos, "dias": int(DIAS_MES[mes]),
                "IDADE_ANOS": ant["IDADE_ANOS"], "SEXO": ant["SEXO"], "NASC": ant["NASC"],
                "MUNIC_RES": ant["MUNIC_RES"], "antes_fim_mes": ant["DT_SAIDA"].is_month_end,
            })
    return pd.DataFrame(linhas)


def lancar_lacunas(acc: Acumulador, lac: pd.DataFrame) -> dict[str, np.ndarray]:
    """Séries de cada grupo apenas com as lacunas (sexo e idade da fatia anterior)."""
    idade = lac["IDADE_ANOS"].to_numpy(dtype=float)
    mascaras = {
        "total": np.ones(len(lac), dtype=bool),
        "masculino": (lac["SEXO"] == "1").to_numpy(),
        "idade_valida": ~np.isnan(idade),
        "60mais": idade >= 60,
    }
    series = {}
    for nome, m in mascaras.items():
        v = np.zeros(N_MESES, dtype=np.int64)
        np.add.at(v, lac["mes"].to_numpy()[m], lac["dias"].to_numpy()[m])
        series[nome] = v
    return series


def relatar_lacunas(f: pd.DataFrame, lac: pd.DataFrame, acc: Acumulador) -> None:
    titulo("Lacunas de longa permanência")
    por_buraco = lac.groupby("buraco").size()
    log(f"Buracos entre fatias consecutivas : {len(por_buraco):,} "
        f"(de um mês: {int((por_buraco == 1).sum()):,})")
    log(f"Lacunas (meses ausentes)          : {len(lac):,}")
    log(f"Pacientes-dia reconstruídos       : {int(lac['dias'].sum()):,}")
    log("\nMeses com mais lacunas:")
    for mes, n in lac["mes"].value_counts().head(5).items():
        dias = int(lac.loc[lac["mes"] == mes, "dias"].sum())
        log(f"  {rotulo_mes(int(mes))}: {n:>4} lacunas, {dias:>6,} pacientes-dia")

    # evidência de continuidade: o que muda entre a fatia anterior e a posterior ao buraco
    pos = lac["buraco"].unique()
    ant, dep = f.loc[pos - 1].reset_index(drop=True), f.loc[pos].reset_index(drop=True)
    log("\nContinuidade entre as fatias vizinhas de cada buraco:")
    log(f"  fatia anterior termina no último dia do mês: "
        f"{int(ant['DT_SAIDA'].dt.is_month_end.sum())} de {len(pos)}")
    log(f"  mesmo DT_INTER registrado                  : "
        f"{int((ant['DT_INTER'] == dep['DT_INTER']).sum())} de {len(pos)}")
    log(f"  mesmo DIAG_PRINC                           : "
        f"{int((ant['DIAG_PRINC'] == dep['DIAG_PRINC']).sum())} de {len(pos)}")

    # limitações: a regra não é alterada; o tamanho de cada caso fica registrado
    fora_fim = lac[~lac["antes_fim_mes"]]
    outra = lac.merge(f[["N_AIH", "mes", "NASC", "SEXO", "MUNIC_RES"]],
                      on=["mes", "NASC", "SEXO", "MUNIC_RES"], suffixes=("", "_outra"))
    outra = outra[outra["N_AIH"] != outra["N_AIH_outra"]].drop_duplicates(["N_AIH", "mes"])
    log("\nLimitações conhecidas (regra mantida; registradas para o texto):")
    log(f"  Lacuna após fatia que não termina no fim do mês (possível saída real): "
        f"{len(fora_fim)} lacunas, {int(fora_fim['dias'].sum())} dias")
    log(f"  Lacuna com fatia de outra AIH de mesmo NASC, SEXO e MUNIC_RES "
        f"(possível troca de AIH): {len(outra)} lacunas, {int(outra['dias'].sum())} dias")

    primeira = f.groupby("N_AIH")["mes"].min().rename("primeira")
    ab = pd.concat(acc.aberturas, ignore_index=True).join(primeira, on="N_AIH")
    faltam = ab["primeira"] - ab["mes"] - 1
    meses = [m for m0, n in zip(ab["mes"], faltam) if n > 0 for m in range(m0 + 1, m0 + 1 + int(n))]
    log(f"  Mês ausente entre IDENT = 1 e a primeira fatia (fora da regra, não reconstruído): "
        f"{int((faltam > 0).sum())} cadeias, {len(meses)} meses, "
        f"{int(DIAS_MES[meses].sum()):,} dias")


# ============================================================
# SÉRIES, AUDITORIA E CONFERÊNCIA
# ============================================================

def montar_serie(acc: Acumulador, lacunas: dict[str, np.ndarray]) -> pd.DataFrame:
    sem = acc.grupos["total"].serie()
    tudo = {nome: g.serie() + lacunas[nome] for nome, g in acc.grupos.items()}
    d = pd.DataFrame({
        "mes": (MES_BASE + np.arange(N_MESES)).astype("datetime64[ms]"),
        "pacientes_dia": tudo["total"],
        "pacientes_dia_sem_reconstrucao": sem,
        "pacientes_dia_zeros1": tudo["total"] + acc.zeros,
        "pd_reconstruidos": lacunas["total"],
        "pd_masculino": tudo["masculino"],
        "pd_idade_valida": tudo["idade_valida"],
        "pd_60mais": tudo["60mais"],
    })
    return d


def auditar(acc: Acumulador, d: pd.DataFrame, lac: pd.DataFrame) -> dict:
    c = acc.cont
    compl = d.iloc[I_JANELA:I_FIM_COMPL]
    model = d.iloc[I_JANELA:I_FIM_MODEL]

    sem = d["pacientes_dia_sem_reconstrucao"].to_numpy()
    diverge = np.nonzero(sem[I_JANELA:I_FIM_COMPL] != acc.independente[I_JANELA:I_FIM_COMPL])[0]
    por_buraco = lac.groupby("buraco").size()

    obtido = {item: c[item] for item in REFERENCIA if item in c}
    obtido.update({
        "auditoria: alocado - DIAS_PERM da base": int(sem.sum()) - c["soma de DIAS_PERM"],
        "auditoria: meses divergentes no método independente": len(diverge),
        "total 01/2015-12/2025, sem reconstrução": int(compl["pacientes_dia_sem_reconstrucao"].sum()),
        "buracos entre fatias": len(por_buraco),
        "buracos de um mês": int((por_buraco == 1).sum()),
        "lacunas": len(lac),
        "pacientes-dia das lacunas": int(lac["dias"].sum()),
        "meses na série completa": len(compl),
        "meses na série de modelagem": len(model),
        "total do alvo (zeros = 0, com reconstrução)": int(model["pacientes_dia"].sum()),
        "total do alvo (zeros = 0, sem reconstrução)": int(model["pacientes_dia_sem_reconstrucao"].sum()),
        "total do alvo (zeros = 1, com reconstrução)": int(model["pacientes_dia_zeros1"].sum()),
        "pacientes-dia com idade inválida": int((d["pacientes_dia"] - d["pd_idade_valida"]).sum()),
    })
    for mes in ["2015-01", "2020-04", "2021-11", "2025-08"]:
        obtido[f"mês {mes}"] = int(d.at[indice_mes(mes), "pacientes_dia"])

    titulo("Auditoria")
    log(f"Soma de DIAS_PERM na base                  : {c['soma de DIAS_PERM']:,}")
    log(f"Alocado em todos os meses, sem reconstrução: {int(sem.sum()):,}")
    log(f"  antes de 01/2015 (fora das séries)       : {int(sem[:I_JANELA].sum()):,}")
    log(f"  de 01/2015 a 12/2025                     : "
        f"{int(compl['pacientes_dia_sem_reconstrucao'].sum()):,}")
    log("\nDecomposição de 01/2015 a 12/2025, sem reconstrução:")
    for item in ["decomposição: IDENT != 5 inteiro na janela até 12/2025",
                 "decomposição: IDENT = 5", "decomposição: parcelas de borda"]:
        log(f"  {item.removeprefix('decomposição: '):<45}: {c[item]:>12,}")
    soma = sum(c[i] for i in REFERENCIA if i.startswith("decomposição"))
    log(f"  {'soma':<45}: {soma:>12,}")
    obtido["auditoria: decomposição - total sem reconstrução"] = (
        soma - obtido["total 01/2015-12/2025, sem reconstrução"])
    log("\nMétodo independente (sobreposição de intervalos), 132 meses: "
        f"{len(diverge)} meses divergentes")

    invalidos = (d["pd_masculino"] > d["pacientes_dia"]) | (d["pd_60mais"] > d["pd_idade_valida"])
    obtido["auditoria: meses com grupo maior que o total"] = int(invalidos.sum())
    obtido["auditoria: meses não consecutivos ou com nulos"] = int(
        (model["mes"].diff().dropna().dt.days > 31).sum() + model.isna().sum().sum())
    return obtido


def relatar_series(d: pd.DataFrame) -> None:
    titulo("Série mensal")
    compl = d.iloc[I_JANELA:I_FIM_COMPL]
    model = d.iloc[I_JANELA:I_FIM_MODEL]
    log(f"Série de modelagem: {model['mes'].iloc[0]:%Y-%m} a {model['mes'].iloc[-1]:%Y-%m}, "
        f"{len(model)} meses")
    log(f"Efeito das saídas no mesmo dia (zeros = 1 - zeros = 0): "
        f"{int((model['pacientes_dia_zeros1'] - model['pacientes_dia']).sum()):,}")

    log("\nEfeito da reconstrução em torno de 11/2021:")
    log(f"  {'mês':<8}{'sem reconstrução':>18}{'com reconstrução':>18}{'acréscimo':>11}")
    for mes in ["2021-09", "2021-10", "2021-11", "2021-12", "2022-01"]:
        lin = d.loc[indice_mes(mes)]
        log(f"  {mes:<8}{lin['pacientes_dia_sem_reconstrucao']:>18,}{lin['pacientes_dia']:>18,}"
            f"{lin['pd_reconstruidos'] / lin['pacientes_dia_sem_reconstrucao']:>11.2%}")

    acrescimo = (compl["pd_reconstruidos"] / compl["pacientes_dia_sem_reconstrucao"]).sort_values()
    log(f"  meses com lacuna: {int((compl['pd_reconstruidos'] > 0).sum())}; "
        f"maiores acréscimos: {acrescimo.iloc[-1]:.2%} e {acrescimo.iloc[-2]:.2%}")

    log("\nTotais da série completa (01/2015 a 12/2025):")
    log(f"  zeros = 0, com reconstrução: {int(compl['pacientes_dia'].sum()):,}")
    log(f"  zeros = 0, sem reconstrução: {int(compl['pacientes_dia_sem_reconstrucao'].sum()):,}")
    log(f"  zeros = 1, com reconstrução: {int(compl['pacientes_dia_zeros1'].sum()):,}")

    log("\nMeses finais da série completa (os posteriores a 08/2025 ficam fora da modelagem):")
    for _, lin in compl.tail(6).iterrows():
        log(f"  {lin['mes']:%Y-%m}: {lin['pacientes_dia']:>9,}")

    tot = model["pacientes_dia"].sum()
    log("\nParticipação na série de modelagem:")
    log(f"  sexo masculino : {model['pd_masculino'].sum() / tot:.2%}")
    log(f"  60 anos ou mais: {model['pd_60mais'].sum() / model['pd_idade_valida'].sum():.2%}")


def conferir(obtido: dict) -> bool:
    titulo("CONFERÊNCIA COM OS VALORES DE REFERÊNCIA")
    esperados = dict(REFERENCIA)
    esperados["auditoria: decomposição - total sem reconstrução"] = 0
    esperados["auditoria: meses com grupo maior que o total"] = 0
    esperados["auditoria: meses não consecutivos ou com nulos"] = 0
    log(f"{'item':<56}{'esperado':>12}{'obtido':>12}  situação")
    tudo_ok = True
    for item, esperado in esperados.items():
        ok = obtido[item] == esperado
        tudo_ok &= ok
        log(f"{item:<56}{esperado:>12,}{obtido[item]:>12,}  {'OK' if ok else 'DIVERGE'}")
    return tudo_ok


def gravar(d: pd.DataFrame, destino: Path) -> None:
    """Grava em arquivo temporário e promove, para não deixar parquet pela metade."""
    temp = destino.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(d, preserve_index=False), temp)
    temp.replace(destino)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not ENTRADA.is_file():
        sys.exit(f"Arquivo de entrada não encontrado: {ENTRADA}. Rodar a Etapa 1 antes.")

    titulo("ETAPA 3 — Variável-alvo")
    log(f"Entrada: {ENTRADA.relative_to(RAIZ)}")

    f = ler_fatias()
    acc = processar(f["N_AIH"].unique())
    lac = achar_lacunas(f)
    relatar_lacunas(f, lac, acc)

    d = montar_serie(acc, lancar_lacunas(acc, lac))
    obtido = auditar(acc, d, lac)
    relatar_series(d)
    tudo_ok = conferir(obtido)

    if tudo_ok:
        gravar(d.iloc[I_JANELA:I_FIM_COMPL].reset_index(drop=True), ARQ_COMPL)
        gravar(d.iloc[I_JANELA:I_FIM_MODEL].reset_index(drop=True), ARQ_SERIE)
        log(f"\nSéries gravadas: {ARQ_SERIE.relative_to(RAIZ)} e {ARQ_COMPL.relative_to(RAIZ)}")
    else:
        log("\nCONFERÊNCIA FALHOU: séries NÃO gravadas. Investigar os itens 'DIVERGE'.")

    ARQ_REL.write_text("\n".join(_linhas), encoding="utf-8")
    print(f"Relatório: {ARQ_REL.relative_to(RAIZ)}")
    sys.exit(0 if tudo_ok else 1)


if __name__ == "__main__":
    main()
