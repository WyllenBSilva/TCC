#!/usr/bin/env python3
"""
valida_corte_janela.py
======================
Valida o corte da janela de modelagem em 08/2025 diretamente a partir da base
limpa, sem usar nenhum relatório anterior.

Uso
---
    python tests/valida_corte_janela.py

Entrada (somente leitura):
    pipeline/saidas/etapa1_base_limpa.parquet

Saídas (em tests/saidas/):
    relatorio_corte_janela.txt
    completude_por_mes_e_defasagem.csv

Ideia central
-------------
Monta-se a matriz P[m, c] = pacientes-dia ocupados no mês m (Abordagem B, sem a
reconstrução das lacunas) trazidos pelos registros da competência c. Com ela, a
base pode ser "rebobinada": a soma de P[m, c] para c <= m + k é o que se
conheceria do mês m se a última competência disponível fosse m + k. A razão
entre essa soma e o total final do mês mede a completude com k competências
posteriores, que é exatamente a situação dos últimos meses da base atual:
    ago/2025: k = 4 | set: k = 3 | out: k = 2 | nov: k = 1 | dez: k = 0
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ    = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "pipeline" / "saidas" / "etapa1_base_limpa.parquet"
SAIDA   = RAIZ / "tests" / "saidas"
ARQ_REL = SAIDA / "relatorio_corte_janela.txt"
ARQ_CSV = SAIDA / "completude_por_mes_e_defasagem.csv"

TAM_LOTE = 250_000
COLUNAS  = ["IDENT", "DT_INTER", "DT_SAIDA", "DIAS_PERM", "ANO_CMPT", "MES_CMPT"]

# Índice de mês = ano * 12 + (mês - 1)
MES_INI, MES_FIM   = 2014 * 12, 2025 * 12 + 11
CMPT_INI, CMPT_FIM = 2015 * 12, 2025 * 12 + 11
N_MES, N_CMPT = MES_FIM - MES_INI + 1, CMPT_FIM - CMPT_INI + 1

ULT_CMPT = CMPT_FIM
EPOCA = np.datetime64("1970-01-01")

MESES_FINAIS = [(2025, 7), (2025, 8), (2025, 9), (2025, 10), (2025, 11), (2025, 12)]


# ============================================================
# UTILITÁRIOS
# ============================================================

_linhas: list[str] = []


def log(texto="") -> None:
    print(texto)
    _linhas.append(str(texto))


def titulo(texto: str) -> None:
    log("\n" + "=" * 78)
    log(texto)
    log("=" * 78)


def idx(ano: int, mes: int) -> int:
    return ano * 12 + mes - 1


def rotulo(i: int) -> str:
    return f"{i // 12}-{i % 12 + 1:02d}"


def inicio_mes(i: int) -> int:
    """Dias desde 1970-01-01 no primeiro dia do mês de índice i."""
    d = np.datetime64(f"{i // 12}-{i % 12 + 1:02d}-01")
    return int((d - EPOCA).astype("int64"))


# ============================================================
# MATRIZES MÊS x COMPETÊNCIA
# ============================================================

def montar_matrizes() -> dict:
    limites = [(inicio_mes(i), inicio_mes(i + 1)) for i in range(MES_INI, MES_FIM + 1)]
    comum = np.zeros((N_MES, N_CMPT), dtype="int64")   # pacientes-dia, IDENT != 5
    longa = np.zeros((N_MES, N_CMPT), dtype="int64")   # pacientes-dia, IDENT = 5
    saidas = np.zeros((N_MES, N_CMPT), dtype="int64")  # registros IDENT != 5 por mês de saída
    soma_dp = 0

    for lote in pq.ParquetFile(ENTRADA).iter_batches(batch_size=TAM_LOTE, columns=COLUNAS):
        ident = lote.column("IDENT").to_numpy()
        dp = lote.column("DIAS_PERM").to_numpy().astype("int64")
        s = lote.column("DT_INTER").to_numpy(zero_copy_only=False).astype("datetime64[D]")
        e = lote.column("DT_SAIDA").to_numpy(zero_copy_only=False).astype("datetime64[D]")
        cmpt = (lote.column("ANO_CMPT").to_numpy().astype("int64") * 12
                + lote.column("MES_CMPT").to_numpy().astype("int64") - 1 - CMPT_INI)
        ms = e.astype("datetime64[M]").astype("int64") + 1970 * 12 - MES_INI
        soma_dp += int(dp.sum())

        is5 = ident == 5
        np.add.at(longa, (ms[is5], cmpt[is5]), dp[is5])
        np.add.at(saidas, (ms[~is5], cmpt[~is5]), 1)

        # IDENT != 5: dias de [DT_INTER, DT_SAIDA) distribuídos pelos meses
        si = (s[~is5] - EPOCA).astype("int64")
        ei = (e[~is5] - EPOCA).astype("int64")
        cc = cmpt[~is5]
        for m, (a, b) in enumerate(limites):
            dias = np.clip(ei, a, b) - np.clip(si, a, b)
            tem = dias > 0
            if tem.any():
                np.add.at(comum[m], cc[tem], dias[tem])

    return {"comum": comum, "longa": longa, "saidas": saidas, "soma_dp": soma_dp}


def ate_k(mat: np.ndarray, mes: int, k: int) -> int:
    """Soma de mat[mes, c] para toda competência c <= mes + k."""
    fim = min(mes + k, ULT_CMPT) - CMPT_INI
    return int(mat[mes - MES_INI, : fim + 1].sum()) if fim >= 0 else 0


def total(mat: np.ndarray, mes: int) -> int:
    return int(mat[mes - MES_INI].sum())


# ============================================================
# ANÁLISES
# ============================================================

def lacunas_por_mes() -> dict:
    """Dias reconstruídos por mês: todo mês sem fatia entre duas fatias IDENT = 5 da mesma AIH."""
    fatias = pq.read_table(ENTRADA, columns=["N_AIH", "DT_SAIDA"], filters=[("IDENT", "=", 5)]).to_pandas()
    fatias["mes"] = fatias["DT_SAIDA"].dt.year * 12 + fatias["DT_SAIDA"].dt.month - 1
    dias: dict[int, int] = {}
    n = 0
    for _, meses in fatias.groupby("N_AIH")["mes"]:
        ordenados = sorted(set(meses))
        for a, b in zip(ordenados, ordenados[1:]):
            for meio in range(a + 1, b):
                dias[meio] = dias.get(meio, 0) + int(pd.Period(rotulo(meio), freq="M").days_in_month)
                n += 1
    dias["n"] = n
    return dias


def auditar(P: np.ndarray, M: dict) -> None:
    titulo("1. Auditoria do cálculo")
    soma = int(P.sum())
    log(f"Soma da matriz de pacientes-dia        : {soma:,}")
    log(f"Soma de DIAS_PERM em todos os registros: {M['soma_dp']:,}")
    log(f"Diferença                              : {soma - M['soma_dp']:,}")

    jan = sum(total(P, m) for m in range(idx(2015, 1), idx(2025, 8) + 1))
    log(f"\nTotal 01/2015-08/2025, sem reconstrução: {jan:,} (referência: 25.421.940)")

    lac = lacunas_por_mes()
    n_lac = lac.pop("n")
    log(f"Lacunas reconstruídas a partir da base : {n_lac:,}, com {sum(lac.values()):,} "
        "pacientes-dia (referência: 298 e 9.018)")
    jan_rec = jan + sum(d for m, d in lac.items() if idx(2015, 1) <= m <= idx(2025, 8))
    log(f"Total 01/2015-08/2025, com reconstrução: {jan_rec:,} (referência: 25.430.958)")
    log("\nMeses de controle:")
    log(f"  {'mês':>8}{'sem reconstr.':>15}{'lacunas':>9}{'com reconstr.':>15}{'referência':>12}")
    for (a, m), ref in {(2015, 1): 177_864, (2020, 4): 137_807,
                        (2021, 11): 194_535, (2025, 8): 243_258}.items():
        mes = idx(a, m)
        d = lac.get(mes, 0)
        log(f"  {rotulo(mes):>8}{total(P, mes):>15,}{d:>9,}{total(P, mes) + d:>15,}{ref:>12,}")

    antes = sum(int(P[m - MES_INI, : max(m - CMPT_INI, 0)].sum())
                for m in range(MES_INI, MES_FIM + 1))
    log(f"\nPacientes-dia em competência anterior ao mês de ocupação: {antes:,}")
    pre = sum(total(P, m) for m in range(MES_INI, idx(2015, 1)))
    log(f"Pacientes-dia ocupados antes de 2015 (fora da janela)   : {pre:,}")


def defasagem_registros(M: dict) -> None:
    titulo("2. Registros IDENT != 5: competência x mês da saída, por ano da saída")
    saidas = M["saidas"]
    ks = [0, 1, 2, 3, 4, 6]
    log(f"{'ano':>6}{'registros':>11}{'lag<0':>8}" + "".join(f"{'k<=' + str(k):>9}" for k in ks))
    for ano in range(2015, 2025):
        ms = range(idx(ano, 1), idx(ano, 12) + 1)
        n = sum(total(saidas, m) for m in ms)
        neg = sum(int(saidas[m - MES_INI, : max(m - CMPT_INI, 0)].sum()) for m in ms)
        cols = "".join(f"{sum(ate_k(saidas, m, k) for m in ms) / n:>9.2%}" for k in ks)
        log(f"{ano:>6}{n:>11,}{neg:>8,}{cols}")

    log("\nEstabilidade recente: saídas de jan a ago de cada ano, como fração do que")
    log("chegou até k = 4 (o máximo disponível para ago/2025):")
    log(f"{'ano':>6}" + "".join(f"{'k<=' + str(k):>9}" for k in [0, 1, 2, 3]))
    for ano in range(2021, 2026):
        ms = range(idx(ano, 1), idx(ano, 8) + 1)
        base = sum(ate_k(saidas, m, 4) for m in ms)
        cols = "".join(f"{sum(ate_k(saidas, m, k) for m in ms) / base:>9.2%}" for k in [0, 1, 2, 3])
        log(f"{ano:>6}{cols}")


def completude_por_k(P: np.ndarray, M: dict) -> dict:
    titulo("3. Completude em PACIENTES-DIA com k competências posteriores")
    log("Completude = pacientes-dia conhecidos com competência <= m + k / total final do mês.")
    log("Meses de referência: 2023-01 a 2024-12 (todos com 12 ou mais competências posteriores).\n")
    ref = range(idx(2023, 1), idx(2024, 12) + 1)
    equivalente = {0: "dez/2025", 1: "nov/2025", 2: "out/2025", 3: "set/2025", 4: "ago/2025",
                   5: "jul/2025", 6: "jun/2025", 7: "mai/2025", 8: "abr/2025"}
    log(f"{'k':>3}  {'situação de':<10}{'déficit médio':>14}{'menor':>9}{'maior':>9}"
        f"{'comum':>9}{'IDENT=5':>9}")
    medias = {}
    for k in [0, 1, 2, 3, 4, 5, 6, 8, 12]:
        defs = np.array([1 - ate_k(P, m, k) / total(P, m) for m in ref])
        d_c = np.mean([1 - ate_k(M["comum"], m, k) / total(M["comum"], m) for m in ref])
        d_l = np.mean([1 - ate_k(M["longa"], m, k) / total(M["longa"], m) for m in ref])
        medias[k] = defs.mean()
        log(f"{k:>3}  {equivalente.get(k, ''):<10}{defs.mean():>14.3%}{defs.min():>9.3%}"
            f"{defs.max():>9.3%}{d_c:>9.3%}{d_l:>9.3%}")

    log("\nMesma medida em meses antigos (2016-01 a 2022-12), com 36 ou mais competências")
    log("posteriores como total final:")
    antigos = range(idx(2016, 1), idx(2022, 12) + 1)
    for k in [2, 3, 4, 5, 6, 12, 24]:
        defs = np.array([1 - ate_k(P, m, k) / total(P, m) for m in antigos])
        log(f"  k = {k:>2}: déficit médio {defs.mean():.3%}  (menor {defs.min():.3%}, maior {defs.max():.3%})")

    log("\nCinco maiores déficits com k = 4 (2016-01 a 2024-12):")
    meses = list(range(idx(2016, 1), idx(2024, 12) + 1))
    defs = sorted(((1 - ate_k(P, m, 4) / total(P, m), m) for m in meses), reverse=True)
    for d, m in defs[:5]:
        log(f"  {rotulo(m)}: {d:.3%}")
    return medias


def rebobinar(P: np.ndarray) -> None:
    titulo("4. Simulação: e se a base tivesse parado na competência de dezembro do ano Y?")
    log("Déficit de cada mês em relação ao seu total final, com a base cortada em 12/Y.")
    log("É a mesma situação da base atual (corte em 12/2025), nos mesmos meses do calendário.\n")
    meses = [6, 7, 8, 9, 10, 11, 12]
    log(f"{'corte':>8}" + "".join(f"{['', 'jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez'][m]:>9}"
                                  for m in meses))
    acum = {m: [] for m in meses}
    for ano in range(2016, 2025):
        cels = []
        for m in meses:
            mes = idx(ano, m)
            d = 1 - ate_k(P, mes, idx(ano, 12) - mes) / total(P, mes)
            acum[m].append(d)
            cels.append(f"{d:>9.2%}")
        log(f"{'12/' + str(ano):>8}" + "".join(cels))
    log(f"{'média':>8}" + "".join(f"{np.mean(acum[m]):>9.2%}" for m in meses))
    log(f"{'pior':>8}" + "".join(f"{np.max(acum[m]):>9.2%}" for m in meses))


def efeito_nos_meses_finais(P: np.ndarray, medias: dict) -> None:
    titulo("5. Tamanho do déficit esperado nos meses finais de 2025, em pacientes-dia")
    log("Valor esperado = observado / (1 - déficit médio de referência para aquele k).\n")
    log(f"{'mês':>8}{'k':>3}{'observado':>12}{'esperado':>12}{'faltam':>10}{'déficit':>9}")
    for a, m in MESES_FINAIS:
        mes = idx(a, m)
        k = ULT_CMPT - mes
        obs = total(P, mes)
        esp = obs / (1 - medias[k])
        log(f"{rotulo(mes):>8}{k:>3}{obs:>12,}{esp:>12,.0f}{esp - obs:>10,.0f}{medias[k]:>9.3%}")

    janela = np.array([total(P, m) for m in range(idx(2015, 1), idx(2025, 8) + 1)], dtype="float64")
    var = np.abs(np.diff(janela))
    log("\nVariação natural da série, para comparação (|y(t) - y(t-1)|, 01/2015 a 08/2025):")
    log(f"  mediana {np.median(var):,.0f}  |  média {var.mean():,.0f}  |  percentil 10 {np.percentile(var, 10):,.0f}")


def metodo_ano_anterior(P: np.ndarray) -> None:
    titulo("6. Conferência independente: comparação com o mesmo mês do ano anterior")
    log("Razão 2025/2024 de cada mês; déficit = 1 - razão / crescimento médio de jan a jul.\n")
    r = np.array([total(P, idx(2025, m)) / total(P, idx(2024, m)) for m in range(1, 13)])
    g = r[:7].mean()
    log(f"Crescimento médio jan-jul/2025: {g:.4f}")
    log(f"{'mês':>5}{'2024':>10}{'2025':>10}{'razão':>8}{'déficit':>9}")
    for m in range(1, 13):
        log(f"{m:>5}{total(P, idx(2024, m)):>10,}{total(P, idx(2025, m)):>10,}"
            f"{r[m - 1]:>8.3f}{1 - r[m - 1] / g:>9.1%}")
    ruido = 1 - r[:7] / g
    log(f"\nFaixa do método em jan-jul (meses sem censura relevante): {ruido.min():.1%} a {ruido.max():.1%}")


def gravar_csv(P: np.ndarray) -> None:
    linhas = []
    for mes in range(idx(2015, 1), idx(2025, 12) + 1):
        lin = {"ano_mes": rotulo(mes), "pacientes_dia": total(P, mes)}
        for k in range(0, 7):
            lin[f"ate_k{k}"] = ate_k(P, mes, k)
        linhas.append(lin)
    pd.DataFrame(linhas).to_csv(ARQ_CSV, index=False)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    M = montar_matrizes()
    P = M["comum"] + M["longa"]

    auditar(P, M)
    defasagem_registros(M)
    medias = completude_por_k(P, M)
    rebobinar(P)
    efeito_nos_meses_finais(P, medias)
    metodo_ano_anterior(P)
    gravar_csv(P)

    ARQ_REL.write_text("\n".join(_linhas), encoding="utf-8")
    print(f"\nRelatório: {ARQ_REL.relative_to(RAIZ)}")
    print(f"CSV      : {ARQ_CSV.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
