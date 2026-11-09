#!/usr/bin/env python3
"""
diagnostico_metodologico.py
===========================
Diagnóstico da base consolidada do SIH-SUS (arquivos RD de SC, 2015-2025)
para embasar as decisões metodológicas sobre a variável-alvo.

Uso
---
    python diagnostico_metodologico.py [pasta_entrada] [pasta_saida]

Padrões:
    pasta_entrada = data_parquet_colunas_reduzidas
    pasta_saida   = diagnostico_metodologico

Colunas lidas (as que existirem em cada arquivo):
    N_AIH, DT_INTER, DT_SAIDA, DIAS_PERM, IDENT, ANO_CMPT, MES_CMPT,
    SEQ_AIH5, MORTE, DIAG_PRINC, ESPEC

Se SEQ_AIH5 não estiver nos seus parquets reduzidos, o script roda sem
ela (o bloco que a usa é pulado). Para incluí-la, adicione "SEQ_AIH5" a
colunas_desejadas em reduz_numero_de_colunas_parquet.py e reprocesse.

Saídas (em pasta_saida)
-----------------------
    relatorio_diagnostico.txt   <- ENVIE ESTE ARQUIVO
    serie_mensal_diagnostico.csv <- envie também (132 linhas)

O script apenas LÊ os dados; não altera nenhum arquivo de entrada.

Checagens
---------
    CHECK 0  Visão geral
    CHECK 1  DIAS_PERM x datas; comportamento das IDENT = 5
    CHECK 2  N_AIH repetida (tipos de repetição, mesmo mês, casos extremos)
    CHECK 3  Defasagem entre mês da saída e competência (censura à direita)
    CHECK 4  DIAS_PERM = 0
    CHECK 5  Registros que cruzam os limites da janela 2015-2025
    CHECK 6  Série mensal: Abordagem A x B, zeros e IDENT = 5
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ENTRADA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data_all_concatenado")
SAIDA   = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("diagnostico_metodologico")

COLUNAS_DESEJADAS = [
    "N_AIH", "DT_INTER", "DT_SAIDA", "DIAS_PERM", "IDENT",
    "ANO_CMPT", "MES_CMPT", "SEQ_AIH5", "MORTE", "DIAG_PRINC", "ESPEC",
]

JANELA_INI = pd.Timestamp("2015-01-01")
JANELA_FIM = pd.Timestamp("2025-12-31")
ANO_INI, ANO_FIM = JANELA_INI.year, JANELA_FIM.year

EPOCA = pd.Timestamp("1970-01-01")


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


def mes_idx(serie_datas: pd.Series) -> pd.Series:
    """Índice inteiro de mês: ano*12 + (mês-1). NaT -> NaN."""
    return serie_datas.dt.year * 12 + serie_datas.dt.month - 1


def rotulo_mes(idx: int) -> str:
    return f"{int(idx) // 12}-{int(idx) % 12 + 1:02d}"


def dias_desde_epoca(serie_datas: pd.Series) -> np.ndarray:
    return (serie_datas.values.astype("datetime64[D]")
            - np.datetime64("1970-01-01")).astype("int64")


def colunas_existentes(df: pd.DataFrame, nomes: list[str]) -> list[str]:
    return [c for c in nomes if c in df.columns]


# ============================================================
# CARGA
# ============================================================

def carregar() -> pd.DataFrame:
    arquivos = sorted(ENTRADA.glob("*.parquet"))
    if not arquivos:
        sys.exit(f"Nenhum .parquet encontrado em: {ENTRADA.resolve()}")

    print(f"Lendo {len(arquivos)} arquivo(s) de {ENTRADA} ...")
    partes = []
    for arquivo in arquivos:
        nomes = pq.read_schema(arquivo).names
        cols = [c for c in COLUNAS_DESEJADAS if c in nomes]
        parte = pd.read_parquet(arquivo, columns=cols)
        parte["ARQUIVO"] = arquivo.stem
        partes.append(parte)
    df = pd.concat(partes, ignore_index=True)
    del partes

    for col in ["DT_INTER", "DT_SAIDA"]:
        df[col] = pd.to_datetime(
            df[col].astype(str).str.strip(), format="%Y%m%d", errors="coerce"
        )

    df["IDENT"] = df["IDENT"].astype(str).str.strip()
    df["DIAS_PERM"] = pd.to_numeric(df["DIAS_PERM"], errors="coerce")

    if "N_AIH" in df.columns:
        df["N_AIH"] = df["N_AIH"].astype(str).str.strip()

    if {"ANO_CMPT", "MES_CMPT"} <= set(df.columns):
        df["CMPT_IDX"] = (
            pd.to_numeric(df["ANO_CMPT"], errors="coerce") * 12
            + pd.to_numeric(df["MES_CMPT"], errors="coerce") - 1
        )
    else:
        df["CMPT_IDX"] = np.nan

    return df


# ============================================================
# CHECK 0 — VISÃO GERAL
# ============================================================

def check_visao_geral(df: pd.DataFrame) -> None:
    titulo("CHECK 0 — Visão geral")
    log(f"Registros            : {len(df):,}")
    log(f"Arquivos (competência): {df['ARQUIVO'].nunique():,} "
        f"({df['ARQUIVO'].min()} ... {df['ARQUIVO'].max()})")
    log(f"Colunas lidas        : {', '.join(c for c in df.columns if c not in ('CMPT_IDX',))}")
    log(f"N_AIH presente       : {'N_AIH' in df.columns}")
    log(f"SEQ_AIH5 presente    : {'SEQ_AIH5' in df.columns}")
    log(f"DT_INTER nulas       : {df['DT_INTER'].isna().sum():,}")
    log(f"DT_SAIDA nulas       : {df['DT_SAIDA'].isna().sum():,}")
    log(f"DIAS_PERM nulos      : {df['DIAS_PERM'].isna().sum():,}")
    log("\nDistribuição de IDENT:")
    log(df["IDENT"].value_counts(dropna=False).to_string())


# ============================================================
# CHECK 1 — DIAS_PERM x DATAS; IDENT = 5
# ============================================================

def check_dias_perm_e_ident5(df: pd.DataFrame) -> None:
    titulo("CHECK 1 — DIAS_PERM x datas; comportamento das IDENT = 5")

    v = df[df["DT_INTER"].notna() & df["DT_SAIDA"].notna()].copy()
    v["DIF"] = (v["DT_SAIDA"] - v["DT_INTER"]).dt.days

    log(f"DT_SAIDA < DT_INTER (todos os IDENT): {(v['DIF'] < 0).sum():,}")

    # ---- 1A: registros comuns -------------------------------------------
    c = v[v["IDENT"] != "5"]
    div = c[c["DIAS_PERM"] != c["DIF"]]
    log("\n[1A] IDENT != 5")
    log(f"  Registros                          : {len(c):,}")
    log(f"  DIAS_PERM == DT_SAIDA - DT_INTER   : {pct(len(c) - len(div), len(c))}")
    log(f"  Divergentes                        : {len(div):,}")
    if len(div):
        log("  Diferença (DIAS_PERM - dif. de datas), valores mais comuns:")
        log((div["DIAS_PERM"] - div["DIF"]).value_counts().head(8).to_string())
        cols = colunas_existentes(div, ["N_AIH", "IDENT", "DT_INTER", "DT_SAIDA", "DIAS_PERM"])
        log("  Amostra de divergentes:")
        log(div[cols].assign(DIF=div["DIF"]).head(10).to_string(index=False))

    # ---- 1B: IDENT = 5 ---------------------------------------------------
    l = v[v["IDENT"] == "5"]
    log(f"\n[1B] IDENT = 5  (registros: {len(l):,})")
    if l.empty:
        return

    igual_dia = l["DIAS_PERM"] == l["DT_SAIDA"].dt.day
    igual_mes = l["DIAS_PERM"] == l["DT_SAIDA"].dt.days_in_month
    fim_mes   = l["DT_SAIDA"].dt.is_month_end
    log(f"  DIAS_PERM == dia do mês de DT_SAIDA        : {pct(igual_dia.sum(), len(l))}")
    log(f"  DIAS_PERM == nº de dias do mês de DT_SAIDA : {pct(igual_mes.sum(), len(l))}")
    log(f"  DT_SAIDA é último dia do mês               : {pct(fim_mes.sum(), len(l))}")
    log(f"  DIAS_PERM > dias do mês de DT_SAIDA        : "
        f"{(l['DIAS_PERM'] > l['DT_SAIDA'].dt.days_in_month).sum():,}")
    log(f"  Intervalo entre datas > 365 dias           : {pct((l['DIF'] > 365).sum(), len(l))}")
    log("  DIAS_PERM (IDENT=5):")
    log(l["DIAS_PERM"].describe().to_string())

    exc = l[~igual_dia]
    log(f"\n  Exceções à hipótese 'DIAS_PERM == dia de DT_SAIDA': {len(exc):,}")
    if len(exc):
        cols = colunas_existentes(exc, ["N_AIH", "DT_INTER", "DT_SAIDA", "DIAS_PERM", "ANO_CMPT", "MES_CMPT"])
        log(exc[cols].head(15).to_string(index=False))


# ============================================================
# CHECK 2 — N_AIH REPETIDA
# ============================================================

def check_n_aih(df: pd.DataFrame) -> None:
    titulo("CHECK 2 — N_AIH repetida")

    if "N_AIH" not in df.columns:
        log("N_AIH ausente nos arquivos. Corrija a lista de colunas "
            "(vírgula após \"N_AIH\") e reprocesse. Check pulado.")
        return

    vc = df["N_AIH"].value_counts()
    rep_ids = vc[vc > 1]
    log(f"Registros                       : {len(df):,}")
    log(f"N_AIH distintos                 : {len(vc):,}")
    log(f"N_AIH com mais de 1 registro    : {len(rep_ids):,}")
    log(f"Registros extras (soma de n-1)  : {int((rep_ids - 1).sum()):,}")
    if rep_ids.empty:
        return

    rep = df[df["N_AIH"].isin(rep_ids.index)].copy()
    rep["IS5"] = rep["IDENT"] == "5"
    rep["MES_SAIDA"] = mes_idx(rep["DT_SAIDA"])

    log("\nIDENT dos registros pertencentes a N_AIH repetidas:")
    log(rep["IDENT"].value_counts().to_string())

    # ---- tipos de grupo ---------------------------------------------------
    g = rep.groupby("N_AIH")["IS5"].agg(soma="sum", n="size")
    todos5 = g["soma"] == g["n"]
    nenhum5 = g["soma"] == 0
    misto = ~(todos5 | nenhum5)
    log("\nTipos de grupo (N_AIH repetida):")
    log(f"  {'tipo':<28}{'grupos':>10}{'registros':>12}{'extras':>10}")
    for nome, mask in [("todos IDENT=5", todos5),
                       ("nenhum IDENT=5", nenhum5),
                       ("misto (5 e outros)", misto)]:
        sub = g[mask]
        log(f"  {nome:<28}{len(sub):>10,}{int(sub['n'].sum()):>12,}{int((sub['n'] - 1).sum()):>10,}")

    # ---- duplicatas exatas ------------------------------------------------
    chave = ["N_AIH", "DT_INTER", "DT_SAIDA", "DIAS_PERM", "IDENT"]
    dup = rep.duplicated(subset=chave, keep="first")
    log(f"\nRegistros repetidos na chave {chave}: {int(dup.sum()):,}")
    if dup.any():
        log(rep.loc[dup, "IDENT"].value_counts().to_string())

    # ---- dois registros no mesmo mês de saída ----------------------------
    dm = rep.duplicated(subset=["N_AIH", "MES_SAIDA"], keep=False)
    log("\nN_AIH com 2+ registros no MESMO mês de DT_SAIDA (risco de dupla contagem):")
    for nome, mask in [("IDENT=5", rep["IS5"]), ("IDENT!=5", ~rep["IS5"])]:
        sel = rep[dm & mask]
        log(f"  {nome:<9}: {len(sel):>8,} registros em {sel['N_AIH'].nunique():>7,} N_AIH")

    # ---- casos extremos ---------------------------------------------------
    log("\nCinco N_AIH mais repetidas:")
    for aih in rep_ids.head(5).index:
        s = rep[rep["N_AIH"] == aih]
        linha = (
            f"  N_AIH {aih}: {len(s)} reg | meses de saída distintos: {s['MES_SAIDA'].nunique()} | "
            f"saída {s['DT_SAIDA'].min():%Y-%m-%d} a {s['DT_SAIDA'].max():%Y-%m-%d} | "
            f"IDENT {sorted(s['IDENT'].unique())} | soma DIAS_PERM: {int(s['DIAS_PERM'].sum()):,} | "
            f"DIAS_PERM==dia(DT_SAIDA): {pct((s['DIAS_PERM'] == s['DT_SAIDA'].dt.day).sum(), len(s))}"
        )
        if "SEQ_AIH5" in s.columns:
            linha += f" | SEQ_AIH5 {s['SEQ_AIH5'].min()}..{s['SEQ_AIH5'].max()}"
        log(linha)

    # ---- repetições que NÃO são IDENT=5 ----------------------------------
    outros = rep[~rep["IS5"]].sort_values(["N_AIH", "DT_SAIDA"])
    log(f"\nAmostra de registros repetidos com IDENT != 5 (total: {len(outros):,}):")
    cols = colunas_existentes(outros, ["N_AIH", "IDENT", "DT_INTER", "DT_SAIDA", "DIAS_PERM", "ARQUIVO"])
    log(outros[cols].head(20).to_string(index=False))

    # ---- SEQ_AIH5 ---------------------------------------------------------
    if "SEQ_AIH5" in df.columns:
        l5 = rep[rep["IS5"]]
        log("\nSEQ_AIH5 nos registros IDENT=5 de N_AIH repetidas:")
        log(l5["SEQ_AIH5"].astype(str).value_counts().head(10).to_string())
        dsq = l5.duplicated(subset=["N_AIH", "SEQ_AIH5"], keep=False)
        log(f"Registros com (N_AIH, SEQ_AIH5) repetido: {int(dsq.sum()):,}")


# ============================================================
# CHECK 3 — DEFASAGEM SAÍDA x COMPETÊNCIA
# ============================================================

def check_defasagem(df: pd.DataFrame) -> None:
    titulo("CHECK 3 — Defasagem entre mês da saída e competência (IDENT != 5)")

    d = df[(df["IDENT"] != "5") & df["DT_SAIDA"].notna() & df["CMPT_IDX"].notna()].copy()
    if d.empty:
        log("Sem dados de competência. Check pulado.")
        return
    d["SAIDA_IDX"] = mes_idx(d["DT_SAIDA"]).astype(int)
    d["LAG"] = (d["CMPT_IDX"] - d["SAIDA_IDX"]).astype(int)

    ref = d[d["DT_SAIDA"].dt.year == 2024]
    base_txt = "saídas em 2024 (ano completo, sem truncamento à direita)"
    if ref.empty:
        ref, base_txt = d, "todos os registros"
    log(f"Base da distribuição: {base_txt} — {len(ref):,} registros")
    log(f"  Competência ANTERIOR ao mês da saída (lag < 0): {pct((ref['LAG'] < 0).sum(), len(ref))}")
    log("  Percentual acumulado de registros apresentados com defasagem <= k meses:")
    for k in [0, 1, 2, 3, 4, 6, 12]:
        log(f"    k = {k:>2}: {pct((ref['LAG'] <= k).sum(), len(ref))}")

    log("\nSaídas (IDENT != 5) por mês de DT_SAIDA — últimos 15 meses da janela "
        "e razão vs. mesmo mês do ano anterior:")
    por_mes = d.groupby("SAIDA_IDX").size()
    ultimo = ANO_FIM * 12 + 11
    linhas = []
    for idx in range(ultimo - 14, ultimo + 1):
        n = int(por_mes.get(idx, 0))
        ant = int(por_mes.get(idx - 12, 0))
        linhas.append(f"  {rotulo_mes(idx)}: {n:>9,}   (ano anterior: {ant:>9,}   razão: "
                      f"{(n / ant if ant else float('nan')):.2f})")
    log("\n".join(linhas))


# ============================================================
# CHECK 4 — DIAS_PERM = 0
# ============================================================

def check_zeros(df: pd.DataFrame) -> None:
    titulo("CHECK 4 — DIAS_PERM = 0")

    z = df[df["DIAS_PERM"] == 0]
    log(f"Registros com DIAS_PERM = 0: {len(z):,} ({pct(len(z), len(df))})")
    log("\nPor IDENT:")
    log(z["IDENT"].value_counts().to_string())

    mesmo_dia = df["DT_INTER"].notna() & (df["DT_INTER"] == df["DT_SAIDA"])
    log(f"\nDT_INTER == DT_SAIDA (todos os registros)        : {int(mesmo_dia.sum()):,}")
    log(f"  entre os zeros, com DT_INTER == DT_SAIDA        : {pct((z['DT_INTER'] == z['DT_SAIDA']).sum(), len(z))}")
    log(f"  entre as saídas no mesmo dia, com DIAS_PERM = 0 : {pct((df.loc[mesmo_dia, 'DIAS_PERM'] == 0).sum(), int(mesmo_dia.sum()))}")
    incons = z[z["DT_INTER"] != z["DT_SAIDA"]]
    log(f"  zeros com datas DIFERENTES (inconsistentes)     : {len(incons):,}")
    if len(incons):
        cols = colunas_existentes(incons, ["N_AIH", "IDENT", "DT_INTER", "DT_SAIDA", "DIAS_PERM"])
        log(incons[cols].head(10).to_string(index=False))

    log("\nProporção de zeros por ano de DT_INTER:")
    por_ano = (df.assign(ZERO=(df["DIAS_PERM"] == 0), ANO=df["DT_INTER"].dt.year)
                 .groupby("ANO")["ZERO"].agg(["sum", "size"]))
    por_ano["proporção"] = (por_ano["sum"] / por_ano["size"]).map("{:.2%}".format)
    log(por_ano.rename(columns={"sum": "zeros", "size": "total"}).to_string())

    if "DIAG_PRINC" in z.columns:
        log("\nDIAG_PRINC (3 primeiros caracteres) mais comuns entre os zeros:")
        log(z["DIAG_PRINC"].astype(str).str[:3].value_counts().head(10).to_string())
    if "MORTE" in z.columns:
        log(f"\nÓbito (MORTE=1) entre os zeros: {pct((z['MORTE'].astype(str).str.strip() == '1').sum(), len(z))}")


# ============================================================
# CHECK 5 — LIMITES DA JANELA
# ============================================================

def check_limites(df: pd.DataFrame) -> None:
    titulo("CHECK 5 — Registros que cruzam os limites da janela 2015-2025")

    antes = df["DT_INTER"] < JANELA_INI
    depois = df["DT_SAIDA"] > JANELA_FIM
    log(f"DT_INTER antes de {JANELA_INI:%Y-%m-%d}: {int(antes.sum()):,}")
    if antes.any():
        log(df.loc[antes, "IDENT"].value_counts().to_string())
        log(f"  menor DT_INTER: {df.loc[antes, 'DT_INTER'].min():%Y-%m-%d}")
    log(f"DT_SAIDA depois de {JANELA_FIM:%Y-%m-%d}: {int(depois.sum()):,}")
    if depois.any():
        log(df.loc[depois, "IDENT"].value_counts().to_string())
        log(f"  maior DT_SAIDA: {df.loc[depois, 'DT_SAIDA'].max():%Y-%m-%d}")


# ============================================================
# CHECK 6 — SÉRIE MENSAL (A x B, ZEROS, IDENT=5)
# ============================================================

def construir_serie(df: pd.DataFrame) -> pd.DataFrame:
    """
    Abordagem A : soma de DIAS_PERM no mês de DT_INTER (todos os IDENT).
    Abordagem B : IDENT != 5  -> dias do intervalo [DT_INTER, DT_SAIDA)
                                 distribuídos pelos meses;
                  IDENT == 5  -> DIAS_PERM no mês de DT_SAIDA.
    """
    meses = list(range(ANO_INI * 12, ANO_FIM * 12 + 12))
    limites = []
    for idx in meses:
        ini = pd.Timestamp(idx // 12, idx % 12 + 1, 1)
        fim = ini + pd.offsets.MonthBegin(1)
        limites.append(((ini - EPOCA).days, (fim - EPOCA).days))

    validos = df[df["DT_INTER"].notna() & df["DT_SAIDA"].notna()
                 & (df["DT_SAIDA"] >= df["DT_INTER"])]

    # ---- B para IDENT != 5: distribuição por mês --------------------------
    comuns = validos[validos["IDENT"] != "5"]
    s = dias_desde_epoca(comuns["DT_INTER"])
    e = dias_desde_epoca(comuns["DT_SAIDA"])
    b_sem5 = []
    for a, b in limites:
        b_sem5.append(int((np.clip(e, a, b) - np.clip(s, a, b)).sum()))

    # ---- B para IDENT == 5: DIAS_PERM no mês de DT_SAIDA ------------------
    l5 = validos[validos["IDENT"] == "5"]
    b5 = l5.groupby(mes_idx(l5["DT_SAIDA"]))["DIAS_PERM"].sum()

    # ---- A: DIAS_PERM no mês de DT_INTER (todos) --------------------------
    a_todos = df[df["DT_INTER"].notna()].groupby(mes_idx(df["DT_INTER"]))["DIAS_PERM"].sum()

    # ---- zeros e contagens ------------------------------------------------
    zeros = df[(df["DIAS_PERM"] == 0) & df["DT_INTER"].notna()].groupby(mes_idx(df["DT_INTER"])).size()
    n_inter = df[(df["IDENT"] != "5") & df["DT_INTER"].notna()].groupby(mes_idx(df["DT_INTER"])).size()
    n_saida = df[(df["IDENT"] != "5") & df["DT_SAIDA"].notna()].groupby(mes_idx(df["DT_SAIDA"])).size()
    n_fatias = l5.groupby(mes_idx(l5["DT_SAIDA"])).size()

    serie = pd.DataFrame(index=meses)
    serie["A_todos_por_DT_INTER"] = a_todos.reindex(meses).fillna(0)
    serie["B_sem_ident5"] = b_sem5
    serie["B_ident5_por_DT_SAIDA"] = b5.reindex(meses).fillna(0)
    serie["B_total_zeros_0"] = serie["B_sem_ident5"] + serie["B_ident5_por_DT_SAIDA"]
    serie["B_total_zeros_1"] = serie["B_total_zeros_0"] + zeros.reindex(meses).fillna(0)
    serie["n_zeros"] = zeros.reindex(meses).fillna(0)
    serie["n_internacoes_sem_ident5"] = n_inter.reindex(meses).fillna(0)
    serie["n_saidas_sem_ident5"] = n_saida.reindex(meses).fillna(0)
    serie["n_registros_ident5"] = n_fatias.reindex(meses).fillna(0)
    serie = serie.astype("int64")
    serie.index = [rotulo_mes(i) for i in meses]
    serie.index.name = "ano_mes"

    # ---- teste de identidade (IDENT != 5, registros inteiramente na janela)
    a0, bL = limites[0][0], limites[-1][1]
    dentro = (s >= a0) & (e <= bL)
    soma_datas = int((e - s)[dentro].sum())
    soma_campo = int(comuns["DIAS_PERM"].to_numpy()[dentro].sum())
    n_div = int((comuns["DIAS_PERM"].to_numpy()[dentro] != (e - s)[dentro]).sum())
    return serie, (int(dentro.sum()), soma_datas, soma_campo, n_div)


def check_serie(df: pd.DataFrame) -> pd.DataFrame:
    titulo("CHECK 6 — Série mensal: Abordagem A x B, zeros e IDENT = 5")
    serie, (n_dentro, soma_datas, soma_campo, n_div) = construir_serie(df)

    log("Teste de identidade (IDENT != 5, registros inteiramente dentro da janela):")
    log(f"  registros                 : {n_dentro:,}")
    log(f"  soma dos dias pelas datas : {soma_datas:,}")
    log(f"  soma de DIAS_PERM         : {soma_campo:,}")
    log(f"  diferença                 : {soma_datas - soma_campo:,}")
    log(f"  registros com DIAS_PERM != diferença de datas: {n_div:,}")

    log("\nTotais por ano (pacientes-dia):")
    anos = serie.copy()
    anos["ano"] = [i[:4] for i in anos.index]
    tab = anos.groupby("ano")[["A_todos_por_DT_INTER", "B_total_zeros_0", "B_total_zeros_1",
                               "B_ident5_por_DT_SAIDA"]].sum()
    tab["B0 - A (%)"] = ((tab["B_total_zeros_0"] - tab["A_todos_por_DT_INTER"])
                         / tab["A_todos_por_DT_INTER"] * 100).round(2)
    tab["zeros=1 vs zeros=0 (%)"] = ((tab["B_total_zeros_1"] - tab["B_total_zeros_0"])
                                     / tab["B_total_zeros_0"] * 100).round(2)
    log(tab.to_string())

    log("\nÚltimos 15 meses da série (B, zeros=0) e contagem de saídas:")
    log(serie[["B_total_zeros_0", "B_ident5_por_DT_SAIDA", "n_saidas_sem_ident5",
               "n_registros_ident5"]].tail(15).to_string())

    return serie


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)

    df = carregar()
    check_visao_geral(df)
    check_dias_perm_e_ident5(df)
    check_n_aih(df)
    check_defasagem(df)
    check_zeros(df)
    check_limites(df)
    serie = check_serie(df)

    serie.to_csv(SAIDA / "serie_mensal_diagnostico.csv")
    (SAIDA / "relatorio_diagnostico.txt").write_text("\n".join(_linhas), encoding="utf-8")

    print("\n" + "-" * 74)
    print(f"Relatório : {(SAIDA / 'relatorio_diagnostico.txt').resolve()}")
    print(f"Série CSV : {(SAIDA / 'serie_mensal_diagnostico.csv').resolve()}")
    print("Envie esses dois arquivos.")


if __name__ == "__main__":
    main()