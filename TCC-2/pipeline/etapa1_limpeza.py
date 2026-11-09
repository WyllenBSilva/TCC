#!/usr/bin/env python3
"""
etapa1_limpeza.py
=================
Etapas 0, 1 e 2 do pipeline: conferência das colunas, tipagem e tratamento de
registros inconsistentes da base consolidada do SIH-SUS (arquivos RD de SC).

Uso
---
    python pipeline/etapa1_limpeza.py

Entrada (somente leitura):
    data_all_concatenado/dados_concatenados.parquet

Saídas (em pipeline/saidas/):
    etapa1_base_limpa.parquet   base tipada, um registro por linha do RD
    etapa1_relatorio.txt        contagens e conferência com os valores de referência

O que é feito
-------------
- strip() nos campos de texto; texto vazio vira nulo;
- datas AAAAMMDD -> timestamp (data inválida vira nulo e é contada);
- N_AIH, IDENT, DIAS_PERM, competência, IDADE, COD_IDADE, MORTE -> inteiros;
- IDADE + COD_IDADE -> IDADE_ANOS (anos completos). Código ou valor inválido é
  sinalizado em IDADE_INVALIDA e IDADE_ANOS fica nulo; o registro NÃO é excluído;
- campos de código (SEXO, ESPEC, DIAG_PRINC, ...) -> category, sem recodificar.

Política de exclusão (Etapa 2)
------------------------------
Só data nula, DIAS_PERM nulo ou negativo ou DT_SAIDA < DT_INTER impedem o cálculo
do alvo. Na base atual todos são zero. Se algum aparecer, ou se qualquer conferência
divergir dos valores de referência, o relatório é gravado mas a base limpa NÃO,
para que a divergência seja investigada antes da etapa seguinte.
Nada é deduplicado: N_AIH repetida são parcelas de longa permanência.

Memória
-------
O WSL tem pouca RAM. A conversão é feita em pyarrow, lote a lote, e cada lote
convertido vai direto para um arquivo temporário. Em memória ficam só os
contadores e o vetor de N_AIH. O temporário vira a base final se tudo conferir.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ     = Path(__file__).resolve().parent.parent
ENTRADA  = RAIZ / "data_all_concatenado" / "dados_concatenados.parquet"
SAIDA    = RAIZ / "pipeline" / "saidas"
ARQ_BASE = SAIDA / "etapa1_base_limpa.parquet"
ARQ_TEMP = SAIDA / "etapa1_base_limpa.parquet.tmp"
ARQ_REL  = SAIDA / "etapa1_relatorio.txt"

TAM_LOTE = 250_000

OBRIGATORIAS = ["N_AIH", "IDENT", "DT_INTER", "DT_SAIDA", "DIAS_PERM", "ANO_CMPT", "MES_CMPT"]

# coluna -> tipo inteiro na base limpa
COLS_INTEIRAS = {
    "N_AIH": pa.int64(), "IDENT": pa.int8(), "ANO_CMPT": pa.int16(), "MES_CMPT": pa.int8(),
    "DIAS_PERM": pa.int32(), "IDADE": pa.int16(), "COD_IDADE": pa.int8(), "MORTE": pa.int8(),
}
COLS_DATA   = ["DT_INTER", "DT_SAIDA", "NASC"]
COLS_CODIGO = ["SEXO", "CAR_INT", "ESPEC", "DIAG_PRINC", "RACA_COR", "MUNIC_RES", "MUNIC_MOV"]

# COD_IDADE -> faixa válida de IDADE. Unidades conferidas contra NASC na base inteira:
#   2 = dias, 3 = meses, 4 = anos, 5 = anos acima de cem (idade = 100 + IDADE)
FAIXA_IDADE = {2: (0, 30), 3: (0, 11), 4: (0, 99), 5: (0, 99)}

ORDEM_SAIDA = [
    "N_AIH", "IDENT", "ANO_CMPT", "MES_CMPT", "DT_INTER", "DT_SAIDA", "DIAS_PERM",
    "NASC", "IDADE", "COD_IDADE", "IDADE_ANOS", "IDADE_INVALIDA",
    "SEXO", "MORTE", "CAR_INT", "ESPEC", "DIAG_PRINC", "RACA_COR", "MUNIC_RES", "MUNIC_MOV",
]

# Valores de referência (CLAUDE.md, obtidos pelos diagnósticos em tests/saidas/)
REFERENCIA = {
    "registros": 5_758_981,
    "competências distintas": 132,
    "N_AIH nulas": 0,
    "IDENT nulos": 0,
    "DT_INTER nulas": 0,
    "DT_SAIDA nulas": 0,
    "DIAS_PERM nulos": 0,
    "DIAS_PERM < 0": 0,
    "DT_SAIDA < DT_INTER": 0,
    "IDENT = 1": 5_728_658,
    "IDENT = 5": 30_323,
    "N_AIH distintas": 5_729_062,
    "N_AIH repetidas": 4_839,
    "registros extras (N_AIH)": 29_919,
    "DIAS_PERM = 0": 644_493,
    "DT_INTER anterior a 2015": 34_660,
    "soma de DIAS_PERM (IDENT = 5)": 858_794,
    "IDENT != 5: DIAS_PERM != DT_SAIDA - DT_INTER": 0,
    "IDENT = 5: DIAS_PERM != dia de DT_SAIDA": 0,
    "DIAS_PERM = 0 com DT_INTER != DT_SAIDA": 0,
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


def pct(parte, total) -> str:
    return f"{parte / total:.2%}" if total else "n/a"


def anos_completos(ref: pd.Series, nasc: pd.Series) -> pd.Series:
    """Idade em anos completos na data ref."""
    antes_do_aniversario = ((ref.dt.month < nasc.dt.month)
                            | ((ref.dt.month == nasc.dt.month) & (ref.dt.day < nasc.dt.day)))
    return ref.dt.year - nasc.dt.year - antes_do_aniversario.astype("int16")


# ============================================================
# ETAPA 0 — COLUNAS
# ============================================================

def checar_colunas() -> list[str]:
    titulo("ETAPA 0 — Colunas")
    if not ENTRADA.is_file():
        sys.exit(f"Arquivo de entrada não encontrado: {ENTRADA}")

    nomes = pq.ParquetFile(ENTRADA).schema_arrow.names
    faltam = [c for c in OBRIGATORIAS if c not in nomes]
    log(f"Arquivo      : {ENTRADA.relative_to(RAIZ)}")
    log(f"Colunas      : {', '.join(nomes)}")
    if faltam:
        log(f"FALTAM colunas obrigatórias: {', '.join(faltam)}")
        ARQ_REL.write_text("\n".join(_linhas), encoding="utf-8")
        sys.exit(1)
    log("Obrigatórias : todas presentes")

    conhecidas = set(COLS_INTEIRAS) | set(COLS_DATA) | set(COLS_CODIGO)
    ignoradas = [c for c in nomes if c not in conhecidas]
    if ignoradas:
        log(f"Ignoradas    : {', '.join(ignoradas)}")
    return [c for c in nomes if c in conhecidas]


# ============================================================
# ETAPA 1 — TIPAGEM (em pyarrow, lote a lote)
# ============================================================

def _texto(a: pa.Array) -> pa.Array:
    """strip(); texto vazio vira nulo."""
    a = pc.utf8_trim_whitespace(a.cast(pa.string()))
    return pc.if_else(pc.equal(a, ""), pa.scalar(None, pa.string()), a)


def _inteiro(a: pa.Array, tipo: pa.DataType) -> pa.Array:
    """Texto ou número -> inteiro (texto não numérico vira nulo)."""
    if pa.types.is_integer(a.type):
        return a.cast(tipo)
    a = _texto(a)
    numerico = pc.match_substring_regex(a, r"^-?\d+$")
    return pc.if_else(numerico, a, pa.scalar(None, pa.string())).cast(tipo)


def _data(a: pa.Array) -> pa.Array:
    """Texto AAAAMMDD -> timestamp (data inválida vira nulo)."""
    a = _texto(a)
    ts = pc.strptime(a, format="%Y%m%d", unit="s", error_is_null=True)
    # strptime aceita datas impossíveis (20150231 vira 03/03); a volta ao texto as descarta
    volta = pc.strftime(ts, format="%Y%m%d")
    # o Parquet não tem resolução de segundos: declara-se o tipo que de fato é gravado
    return pc.if_else(pc.equal(volta, a), ts, pa.scalar(None, ts.type)).cast(pa.timestamp("ms"))


def idade_em_anos(idade: pa.Array, cod: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """IDADE + COD_IDADE -> (anos completos, máscara de validade)."""
    i = idade.fill_null(-1).to_numpy().astype("int16")
    c = cod.fill_null(-1).to_numpy().astype("int8")

    valida = np.zeros(len(i), dtype=bool)
    for codigo, (minimo, maximo) in FAIXA_IDADE.items():
        valida |= (c == codigo) & (i >= minimo) & (i <= maximo)

    # anos completos: menos de um ano (dias ou meses) vira 0
    anos = np.select([c == 2, c == 3, c == 4, c == 5], [0, i // 12, i, 100 + i], -1)
    return anos.astype("int16"), valida


def converter_lote(lote: pa.RecordBatch) -> pa.Table:
    cols = {}
    for nome in lote.schema.names:
        a = lote.column(nome)
        if nome in COLS_INTEIRAS:
            cols[nome] = _inteiro(a, COLS_INTEIRAS[nome])
        elif nome in COLS_DATA:
            cols[nome] = _data(a)
        elif nome in COLS_CODIGO:
            cols[nome] = pc.dictionary_encode(_texto(a))

    if "IDADE" in cols and "COD_IDADE" in cols:
        anos, valida = idade_em_anos(cols["IDADE"], cols["COD_IDADE"])
        cols["IDADE_ANOS"] = pa.array(anos, type=pa.int16(), mask=~valida)
        cols["IDADE_INVALIDA"] = pa.array(~valida)

    return pa.table({c: cols[c] for c in ORDEM_SAIDA if c in cols})


# ============================================================
# ACUMULAÇÃO DAS CONTAGENS (lote a lote)
# ============================================================

COLS_CONTAGEM = ["N_AIH", "IDENT", "ANO_CMPT", "MES_CMPT", "DT_INTER", "DT_SAIDA",
                 "DIAS_PERM", "NASC", "IDADE", "COD_IDADE", "IDADE_ANOS", "IDADE_INVALIDA"]


class Acumulador:
    def __init__(self) -> None:
        self.cont = Counter()
        self.minimo: dict = {}
        self.maximo: dict = {}
        self.cod_idade: dict = {}          # código -> [registros, IDADE mín, IDADE máx]
        self.amostra_invalida: list[pd.DataFrame] = []
        self.competencias: set[int] = set()
        self.n_aih: list[np.ndarray] = []
        self.idade_anos: list[pa.ChunkedArray] = []
        self.schema: pa.Schema | None = None

    def extremos(self, nome: str, s: pd.Series) -> None:
        s = s.dropna()
        if len(s):
            self.minimo[nome] = min(self.minimo.get(nome, s.min()), s.min())
            self.maximo[nome] = max(self.maximo.get(nome, s.max()), s.max())

    def adicionar(self, t: pa.Table) -> None:
        c = self.cont
        c["registros"] += t.num_rows
        for nome in t.column_names:
            c[f"nulos {nome}"] += t[nome].null_count
        self.n_aih.append(t["N_AIH"].fill_null(0).to_numpy())
        if "IDADE_ANOS" in t.column_names:
            self.idade_anos.append(t["IDADE_ANOS"])

        d = t.select([x for x in COLS_CONTAGEM if x in t.column_names]).to_pandas()
        is5 = d["IDENT"] == 5
        dif = (d["DT_SAIDA"] - d["DT_INTER"]).dt.days
        dp = d["DIAS_PERM"]

        self.competencias |= set((d["ANO_CMPT"] * 12 + d["MES_CMPT"]).dropna().astype(int))
        c["DT_SAIDA < DT_INTER"] += int((dif < 0).sum())
        c["DIAS_PERM < 0"] += int((dp < 0).sum())
        c["IDENT = 1"] += int((d["IDENT"] == 1).sum())
        c["IDENT = 5"] += int(is5.sum())
        c["DIAS_PERM = 0"] += int((dp == 0).sum())
        c["DT_INTER anterior a 2015"] += int((d["DT_INTER"] < pd.Timestamp("2015-01-01")).sum())
        antigas = d["DT_INTER"] < pd.Timestamp("2014-01-01")
        c["DT_INTER anterior a 2014"] += int(antigas.sum())
        c["DT_INTER anterior a 2014 (IDENT = 5)"] += int((antigas & is5).sum())
        c["DT_INTER = 2008-01-01"] += int((d["DT_INTER"] == pd.Timestamp("2008-01-01")).sum())
        c["soma de DIAS_PERM (IDENT = 5)"] += int(dp[is5].fillna(0).astype("int64").sum())
        c["IDENT = 5 com DT_SAIDA - DT_INTER > 365"] += int((is5 & (dif > 365)).sum())
        c["IDENT != 5: DIAS_PERM != DT_SAIDA - DT_INTER"] += int((~is5 & (dp != dif)).sum())
        c["IDENT = 5: DIAS_PERM != dia de DT_SAIDA"] += int(
            (is5 & (dp != d["DT_SAIDA"].dt.day)).sum())
        c["DIAS_PERM = 0 com DT_INTER != DT_SAIDA"] += int(
            ((dp == 0) & (d["DT_INTER"] != d["DT_SAIDA"])).sum())

        self.extremos("DIAS_PERM (IDENT = 5)", dp[is5])
        for nome in COLS_DATA:
            if nome in d.columns:
                self.extremos(nome, d[nome])

        if "IDADE_ANOS" in d.columns:
            self._adicionar_idade(d, is5)

    def _adicionar_idade(self, d: pd.DataFrame, is5: pd.Series) -> None:
        c = self.cont
        for cod, g in d.groupby("COD_IDADE", dropna=False)["IDADE"]:
            n, mi, ma = self.cod_idade.get(cod, [0, g.min(), g.max()])
            self.cod_idade[cod] = [n + len(g), min(mi, g.min()), max(ma, g.max())]

        invalida = d["IDADE_INVALIDA"]
        c["idade inválida"] += int(invalida.sum())
        faltam = 10 - sum(len(x) for x in self.amostra_invalida)
        if invalida.any() and faltam > 0:
            cols = [x for x in ["N_AIH", "IDENT", "IDADE", "COD_IDADE", "NASC", "DT_INTER",
                                "DIAS_PERM"] if x in d.columns]
            self.amostra_invalida.append(d.loc[invalida, cols].head(faltam))

        # validação cruzada com NASC
        # IDENT != 5: idade na data da internação.
        # IDENT  = 5: idade no último dia do mês anterior a DT_SAIDA (fim da
        #             competência anterior); o DT_INTER dessas fatias não é usado.
        if "NASC" not in d.columns:
            return
        ref = d["DT_INTER"].where(
            ~is5, d["DT_SAIDA"] - pd.to_timedelta(d["DT_SAIDA"].dt.day, unit="D"))
        calc = anos_completos(ref, d["NASC"])
        comparavel = ~invalida & d["NASC"].notna() & ref.notna()
        diverge = comparavel & (d["IDADE_ANOS"] != calc)
        for rot, mask in [("IDENT != 5", ~is5), ("IDENT = 5", is5)]:
            c[f"NASC comparados {rot}"] += int((comparavel & mask).sum())
            c[f"NASC divergentes {rot}"] += int((diverge & mask).sum())
        c["NASC posterior a DT_INTER (IDENT != 5)"] += int(
            (~is5 & (d["NASC"] > d["DT_INTER"])).sum())
        c["NASC = 1899-12-30"] += int((d["NASC"] == pd.Timestamp("1899-12-30")).sum())

        # confere IDADE na unidade do próprio código: distingue 0 dias de 11 meses,
        # que em anos completos são ambos 0
        meses = ((ref.dt.year - d["NASC"].dt.year) * 12 + ref.dt.month - d["NASC"].dt.month
                 - (ref.dt.day < d["NASC"].dt.day).astype("int16"))
        por_codigo = {2: (ref - d["NASC"]).dt.days, 3: meses, 4: calc, 5: calc - 100}
        for cod, valor in por_codigo.items():
            do_codigo = comparavel & (d["COD_IDADE"] == cod)
            c[f"unidade comparados {cod}"] += int(do_codigo.sum())
            c[f"unidade divergentes {cod}"] += int((do_codigo & (d["IDADE"] != valor)).sum())


def processar(colunas: list[str]) -> Acumulador:
    """Lê, converte e grava a base no arquivo temporário, acumulando as contagens."""
    acc = Acumulador()
    escritor = None
    try:
        for lote in pq.ParquetFile(ENTRADA).iter_batches(batch_size=TAM_LOTE, columns=colunas):
            t = converter_lote(lote)
            if escritor is None:
                acc.schema = t.schema
                escritor = pq.ParquetWriter(ARQ_TEMP, t.schema)
            escritor.write_table(t)
            acc.adicionar(t)
            print(f"  ... {acc.cont['registros']:,} linhas convertidas", end="\r", flush=True)
    finally:
        if escritor is not None:
            escritor.close()
    print()
    return acc


# ============================================================
# RELATÓRIO
# ============================================================

def relatar_tipagem(acc: Acumulador) -> None:
    titulo("ETAPA 1 — Tipagem")
    log(f"Registros lidos: {acc.cont['registros']:,}")
    log("\nTipo na base limpa e nulos após a conversão "
        "(texto vazio, número ilegível ou data inválida):")
    for campo in acc.schema:
        log(f"  {campo.name:<15} {str(campo.type):<32} {acc.cont[f'nulos {campo.name}']:>10,}")
    log("\nIntervalo das datas:")
    for nome in COLS_DATA:
        if nome in acc.minimo:
            log(f"  {nome:<9}: {acc.minimo[nome]:%Y-%m-%d} a {acc.maximo[nome]:%Y-%m-%d}")


def relatar_idade(acc: Acumulador) -> None:
    titulo("Idade em anos (IDADE + COD_IDADE)")
    if not acc.idade_anos:
        log("IDADE ou COD_IDADE ausente; conversão pulada.")
        return

    log("COD_IDADE (2 = dias, 3 = meses, 4 = anos, 5 = acima de cem anos):")
    log(f"  {'código':>6}{'registros':>12}{'IDADE mín':>11}{'IDADE máx':>11}")
    for cod in sorted(acc.cod_idade, key=lambda x: -1 if pd.isna(x) else x):
        n, mi, ma = acc.cod_idade[cod]
        log(f"  {str(cod):>6}{n:>12,}{mi:>11}{ma:>11}")

    log(f"\nIdade inválida (sinalizada em IDADE_INVALIDA, registro mantido): "
        f"{acc.cont['idade inválida']:,}")
    if acc.amostra_invalida:
        log(pd.concat(acc.amostra_invalida).to_string(index=False))

    anos = pa.chunked_array([ch for x in acc.idade_anos for ch in x.chunks]).to_pandas()
    log("\nIDADE_ANOS:")
    log(anos.describe().to_string())
    log(f"  0 anos (menos de um ano)       : {int((anos == 0).sum()):,}")
    log(f"  100 anos ou mais               : {int((anos >= 100).sum()):,}")
    log(f"  mais de 110 anos               : {int((anos > 110).sum()):,}")
    log(f"  NASC = 1899-12-30 (registros)  : {acc.cont['NASC = 1899-12-30']:,}")

    if acc.cont["NASC comparados IDENT != 5"] or acc.cont["NASC comparados IDENT = 5"]:
        log("\nValidação cruzada IDADE_ANOS x NASC:")
        for rot, ref in [("IDENT != 5", "ref.: DT_INTER"),
                         ("IDENT = 5", "ref.: fim do mês anterior a DT_SAIDA")]:
            n = acc.cont[f"NASC comparados {rot}"]
            dv = acc.cont[f"NASC divergentes {rot}"]
            log(f"  {rot:<10} ({ref:<37}) comparados {n:>10,}  divergentes {dv:>6,}  "
                f"concordância {pct(n - dv, n)}")
        log(f"  NASC posterior a DT_INTER (IDENT != 5): "
            f"{acc.cont['NASC posterior a DT_INTER (IDENT != 5)']:,}")

        log("\nValidação de IDADE na unidade do código, contra NASC:")
        for cod, unidade in [(2, "dias"), (3, "meses completos"), (4, "anos completos"),
                             (5, "anos completos - 100")]:
            log(f"  COD_IDADE = {cod} ({unidade:<20}) comparados "
                f"{acc.cont[f'unidade comparados {cod}']:>10,}  divergentes "
                f"{acc.cont[f'unidade divergentes {cod}']:>6,}")


def contar(acc: Acumulador) -> dict:
    """Contagens comparadas com REFERENCIA."""
    aih = np.concatenate(acc.n_aih)
    acc.n_aih = []
    _, n = np.unique(aih, return_counts=True)
    del aih
    rep = n[n > 1]

    c = acc.cont
    obtido = {item: c[item] for item in REFERENCIA if item in c}
    obtido.update({
        "registros": c["registros"],
        "competências distintas": len(acc.competencias),
        "N_AIH nulas": c["nulos N_AIH"],
        "IDENT nulos": c["nulos IDENT"],
        "DT_INTER nulas": c["nulos DT_INTER"],
        "DT_SAIDA nulas": c["nulos DT_SAIDA"],
        "DIAS_PERM nulos": c["nulos DIAS_PERM"],
        "N_AIH distintas": len(n),
        "N_AIH repetidas": len(rep),
        "registros extras (N_AIH)": int((rep - 1).sum()),
    })
    return obtido


def relatar_inconsistencias(acc: Acumulador, obtido: dict) -> None:
    titulo("ETAPA 2 — Registros inconsistentes")
    log("Critérios de exclusão (impedem o cálculo do alvo):")
    for item in ["DT_INTER nulas", "DT_SAIDA nulas", "DIAS_PERM nulos", "DIAS_PERM < 0",
                 "DT_SAIDA < DT_INTER"]:
        log(f"  {item:<22}: {obtido[item]:,}")
    log("Registros excluídos: 0 (se algum critério acima for diferente de zero, a base "
        "não é gravada)")

    c = acc.cont
    log("\nCasos preservados (NÃO são inconsistências):")
    log(f"  N_AIH repetidas (parcelas de longa permanência): {obtido['N_AIH repetidas']:,} "
        f"N_AIH, {obtido['registros extras (N_AIH)']:,} registros extras")
    log(f"  DIAS_PERM = 0 (saída no dia da internação)     : {obtido['DIAS_PERM = 0']:,} "
        f"({pct(obtido['DIAS_PERM = 0'], c['registros'])})")
    n_longa = c["IDENT = 5 com DT_SAIDA - DT_INTER > 365"]
    log(f"  IDENT = 5 com DT_SAIDA - DT_INTER > 365 dias   : {n_longa:,} "
        f"({pct(n_longa, c['IDENT = 5'])} das fatias)")
    log(f"  DT_INTER anterior a 2015                       : {c['DT_INTER anterior a 2015']:,}")
    log(f"  DT_INTER anterior a 2014                       : {c['DT_INTER anterior a 2014']:,} "
        f"(IDENT = 5: {c['DT_INTER anterior a 2014 (IDENT = 5)']:,})")
    log(f"  DT_INTER = 2008-01-01 (data-sentinela)         : {c['DT_INTER = 2008-01-01']:,}")
    if "DIAS_PERM (IDENT = 5)" in acc.minimo:
        log(f"  DIAS_PERM nas fatias IDENT = 5                 : "
            f"{int(acc.minimo['DIAS_PERM (IDENT = 5)'])} a "
            f"{int(acc.maximo['DIAS_PERM (IDENT = 5)'])}")


def conferir(obtido: dict) -> bool:
    titulo("CONFERÊNCIA COM OS VALORES DE REFERÊNCIA")
    log(f"{'item':<46}{'esperado':>13}{'obtido':>13}  situação")
    tudo_ok = True
    for item, esperado in REFERENCIA.items():
        ok = obtido[item] == esperado
        tudo_ok &= ok
        log(f"{item:<46}{esperado:>13,}{obtido[item]:>13,}  {'OK' if ok else 'DIVERGE'}")
    return tudo_ok


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    assert ENTRADA.parent not in ARQ_BASE.parents, "a saída não pode ficar em data_all_concatenado/"

    colunas = checar_colunas()
    acc = processar(colunas)
    relatar_tipagem(acc)
    relatar_idade(acc)
    obtido = contar(acc)
    relatar_inconsistencias(acc, obtido)
    tudo_ok = conferir(obtido)

    if tudo_ok:
        ARQ_TEMP.replace(ARQ_BASE)
        meta = pq.ParquetFile(ARQ_BASE).metadata
        log(f"\nBase limpa gravada: {ARQ_BASE.relative_to(RAIZ)} "
            f"({meta.num_rows:,} registros, {meta.num_columns} colunas)")
    else:
        ARQ_TEMP.unlink(missing_ok=True)
        log("\nCONFERÊNCIA FALHOU: base limpa NÃO gravada. Investigar os itens 'DIVERGE'.")

    ARQ_REL.write_text("\n".join(_linhas), encoding="utf-8")
    print(f"Relatório: {ARQ_REL.relative_to(RAIZ)}")
    sys.exit(0 if tudo_ok else 1)


if __name__ == "__main__":
    main()
