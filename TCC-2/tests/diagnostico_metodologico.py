#!/usr/bin/env python3
"""
diagnostico_metodologico.py
===========================
Diagnóstico da base consolidada do SIH-SUS (arquivos RD de SC, 2015-2025)
para embasar as decisões metodológicas sobre a variável-alvo.

Uso
---
    python diagnostico_metodologico.py [pasta_entrada] [pasta_saida]

(pasta_entrada pode ser uma pasta com .parquet ou um único arquivo .parquet)

Padrões:
    pasta_entrada = data_parquet_colunas_reduzidas
    pasta_saida   = diagnostico_metodologico

Colunas lidas (as que existirem em cada arquivo):
    N_AIH, DT_INTER, DT_SAIDA, DIAS_PERM, IDENT, ANO_CMPT, MES_CMPT,
    SEQ_AIH5, MORTE, DIAG_PRINC

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

import gc
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
    "ANO_CMPT", "MES_CMPT", "SEQ_AIH5", "MORTE", "DIAG_PRINC",
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
# CARGA (em lotes, com tipos compactos para economizar memória)
# ============================================================

TAM_LOTE = 500_000
_NULOS = {"N_AIH": 0}


def _para_data(s: pd.Series) -> pd.Series:
    """Converte para datetime64[ns] aceitando texto/número AAAAMMDD ou data nativa."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.astype("datetime64[ns]")
    if pd.api.types.is_numeric_dtype(s):
        txt = s.astype("Int64").astype("string")
    else:
        txt = s.astype("string").str.strip()
    out = pd.to_datetime(txt, format="%Y%m%d", errors="coerce")
    if out.isna().all() and s.notna().any():
        out = pd.to_datetime(s, errors="coerce")
    return out.astype("datetime64[ns]")


def _compactar(parte: pd.DataFrame) -> pd.DataFrame:
    """Converte um lote para tipos numéricos/datas (sem strings Python)."""
    out = {}
    if "N_AIH" in parte.columns:
        n = pd.to_numeric(parte["N_AIH"], errors="coerce")
        _NULOS["N_AIH"] += int(n.isna().sum())
        out["N_AIH"] = n.fillna(0).astype("int64")      # 0 = ausente
    out["DT_INTER"] = _para_data(parte["DT_INTER"])
    out["DT_SAIDA"] = _para_data(parte["DT_SAIDA"])
    out["DIAS_PERM"] = pd.to_numeric(parte["DIAS_PERM"], errors="coerce").astype("float64")
    out["IDENT"] = pd.to_numeric(parte["IDENT"], errors="coerce").fillna(-1).astype("int8")
    if {"ANO_CMPT", "MES_CMPT"} <= set(parte.columns):
        out["CMPT_IDX"] = (pd.to_numeric(parte["ANO_CMPT"], errors="coerce") * 12
                           + pd.to_numeric(parte["MES_CMPT"], errors="coerce") - 1)
    else:
        out["CMPT_IDX"] = pd.Series(np.nan, index=parte.index)
    if "SEQ_AIH5" in parte.columns:
        out["SEQ_AIH5"] = pd.to_numeric(parte["SEQ_AIH5"], errors="coerce").astype("float32")
    if "MORTE" in parte.columns:
        out["MORTE"] = pd.to_numeric(parte["MORTE"], errors="coerce").fillna(-1).astype("int8")
    if "DIAG_PRINC" in parte.columns:
        out["DIAG_PRINC"] = (parte["DIAG_PRINC"].astype("string").str.strip()
                             .str[:3].astype("string[pyarrow]"))
    return pd.DataFrame(out)


def carregar() -> pd.DataFrame:
    if ENTRADA.is_file():
        arquivos = [ENTRADA]
    else:
        arquivos = sorted(ENTRADA.glob("*.parquet"))
    if not arquivos:
        sys.exit(f"Nenhum .parquet encontrado em: {ENTRADA.resolve()}")

    print(f"Lendo {len(arquivos)} arquivo(s) de {ENTRADA} ...")
    partes = []
    total = 0
    for arquivo in arquivos:
        pf = pq.ParquetFile(arquivo)
        cols = [c for c in COLUNAS_DESEJADAS if c in pf.schema_arrow.names]
        for lote in pf.iter_batches(batch_size=TAM_LOTE, columns=cols):
            partes.append(_compactar(lote.to_pandas()))
            total += len(partes[-1])
            print(f"  ... {total:,} linhas lidas", end="\r", flush=True)
    print()
    df = pd.concat(partes, ignore_index=True)
    del partes
    gc.collect()
    print(f"Memória do DataFrame: {df.memory_usage(deep=True).sum() / 1e6:,.0f} MB")
    return df


# ============================================================
# CHECK 0 — VISÃO GERAL
# ============================================================

def check_visao_geral(df: pd.DataFrame) -> None:
    titulo("CHECK 0 — Visão geral")
    log(f"Registros            : {len(df):,}")
    cm = df["CMPT_IDX"].dropna()
    if len(cm):
        log(f"Competências distintas: {cm.nunique():,} "
            f"({rotulo_mes(cm.min())} ... {rotulo_mes(cm.max())})")
    log(f"N_AIH ausentes/ilegíveis (gravadas como 0): {_NULOS['N_AIH']:,}")
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

    v = df.loc[df["DT_INTER"].notna() & df["DT_SAIDA"].notna(),
               colunas_existentes(df, ["N_AIH", "IDENT", "DT_INTER", "DT_SAIDA",
                                       "DIAS_PERM", "CMPT_IDX"])].copy()
    v["DIF"] = (v["DT_SAIDA"] - v["DT_INTER"]).dt.days

    log(f"DT_SAIDA < DT_INTER (todos os IDENT): {(v['DIF'] < 0).sum():,}")

    # ---- 1A: registros comuns -------------------------------------------
    c = v[v["IDENT"] != 5]
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
    l = v[v["IDENT"] == 5]
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
        cols = colunas_existentes(exc, ["N_AIH", "DT_INTER", "DT_SAIDA", "DIAS_PERM", "CMPT_IDX"])
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

    vc = df["N_AIH"].value_counts().drop(0, errors="ignore")
    rep_ids = vc[vc > 1]
    log(f"Registros                       : {len(df):,}")
    log(f"N_AIH distintos                 : {len(vc):,}")
    log(f"N_AIH com mais de 1 registro    : {len(rep_ids):,}")
    log(f"Registros extras (soma de n-1)  : {int((rep_ids - 1).sum()):,}")
    if rep_ids.empty:
        return

    rep = df[df["N_AIH"].isin(rep_ids.index)].copy()
    rep["IS5"] = rep["IDENT"] == 5
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
            f"IDENT {sorted(int(x) for x in s['IDENT'].unique())} | "
            f"soma DIAS_PERM: {int(s['DIAS_PERM'].sum()):,} | "
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

    d = df.loc[(df["IDENT"] != 5) & df["DT_SAIDA"].notna() & df["CMPT_IDX"].notna(),
               ["DT_SAIDA", "CMPT_IDX"]].copy()
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
    por_ano = (pd.DataFrame({"ZERO": df["DIAS_PERM"] == 0, "ANO": df["DT_INTER"].dt.year})
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

    # --- as internações muito antigas são reais ou datas-sentinela? -------
    ant = df.loc[antes & (df["DT_INTER"] < pd.Timestamp("2014-01-01")),
                 ["IDENT", "DT_INTER", "DT_SAIDA", "DIAS_PERM"]]
    log(f"\nRegistros com DT_INTER anterior a 2014 (suspeita de data-sentinela): {len(ant):,}")
    if len(ant):
        log("  Por ano de DT_INTER e IDENT:")
        log(pd.crosstab(ant["DT_INTER"].dt.year, ant["IDENT"]).to_string())
        log("\n  Datas de internação exatas mais frequentes:")
        vc = ant["DT_INTER"].dt.strftime("%Y-%m-%d").value_counts().head(10)
        log("    " + "\n    ".join(f"{d}: {n:,}" for d, n in vc.items()))
        log("\n  Dia 1º do mês entre essas internações: "
            f"{pct((ant['DT_INTER'].dt.day == 1).sum(), len(ant))}")
        log("  Permanência implícita (DT_SAIDA - DT_INTER), em dias:")
        log((ant["DT_SAIDA"] - ant["DT_INTER"]).dt.days.describe().to_string())
        log("\n  Amostra:")
        log(ant.sort_values("DT_INTER").head(10).to_string(index=False))


# ============================================================
# CHECK 6 — SÉRIE MENSAL (A x B, ZEROS, IDENT=5)
# ============================================================
#
# Trabalha apenas com inteiros compactos. O DataFrame original é
# liberado antes de montar a série, para manter o pico de memória baixo.

SENTINELA = -(10 ** 8)


def _dias_int32(serie: pd.Series) -> np.ndarray:
    """Dias desde 1970-01-01 como int32; NaT -> SENTINELA."""
    v = serie.to_numpy("datetime64[D]")
    out = np.full(len(v), SENTINELA, dtype="int32")
    ok = ~pd.isna(v)
    out[ok] = (v[ok] - np.datetime64("1970-01-01")).astype("int64").astype("int32")
    return out


def _mes_int32(serie: pd.Series) -> np.ndarray:
    """Índice de mês (ano*12 + mês-1) como int32; NaT -> -1."""
    a = serie.dt.year.to_numpy("float64")
    m = serie.dt.month.to_numpy("float64")
    idx = a * 12 + m - 1
    return np.where(np.isnan(idx), -1, idx).astype("int32")


def compactar_para_serie(df: pd.DataFrame) -> dict:
    """Extrai só os vetores necessários ao CHECK 6."""
    d = {
        "ident": df["IDENT"].to_numpy("int8"),
        "dp": df["DIAS_PERM"].fillna(-1).to_numpy("int32"),
        "mi": _mes_int32(df["DT_INTER"]),
        "ms": _mes_int32(df["DT_SAIDA"]),
        "s": _dias_int32(df["DT_INTER"]),
        "e": _dias_int32(df["DT_SAIDA"]),
    }
    print(f"Vetores do CHECK 6: {sum(v.nbytes for v in d.values()) / 1e6:,.0f} MB")
    return d


def _soma_por_mes(idx: np.ndarray, pesos: np.ndarray | None, meses: np.ndarray) -> np.ndarray:
    """Soma (ou conta) por índice de mês, restrito à janela."""
    base, n = int(meses[0]), len(meses)
    ok = (idx >= base) & (idx < base + n)
    if pesos is not None:
        ok &= pesos >= 0
    pos = (idx[ok] - base).astype("int64")
    w = pesos[ok].astype("int64") if pesos is not None else None
    return np.bincount(pos, weights=w, minlength=n)[:n]


def construir_serie(d: dict):
    meses = np.arange(ANO_INI * 12, ANO_FIM * 12 + 12, dtype="int64")
    limites = []
    for idx in meses:
        ini = pd.Timestamp(int(idx) // 12, int(idx) % 12 + 1, 1)
        fim = ini + pd.offsets.MonthBegin(1)
        limites.append(((ini - EPOCA).days, (fim - EPOCA).days))

    ident, dp, mi, ms, s, e = (d["ident"], d["dp"], d["mi"], d["ms"], d["s"], d["e"])

    # datas válidas e saída não anterior à internação
    val = (s > SENTINELA) & (e > SENTINELA) & (e >= s)
    comum = val & (ident != 5)
    longa = val & (ident == 5)

    # ---- B, IDENT != 5: dias do intervalo [DT_INTER, DT_SAIDA) por mês ----
    sc, ec = s[comum], e[comum]
    b_sem5 = np.empty(len(meses), dtype="int64")
    for k, (a, b) in enumerate(limites):
        b_sem5[k] = int((np.clip(ec, a, b) - np.clip(sc, a, b)).sum())

    # ---- demais componentes --------------------------------------------
    b5      = _soma_por_mes(ms[longa], dp[longa], meses)
    a_todos = _soma_por_mes(mi, dp, meses)
    zeros   = _soma_por_mes(mi[dp == 0], None, meses)
    n_inter = _soma_por_mes(mi[ident != 5], None, meses)
    n_saida = _soma_por_mes(ms[ident != 5], None, meses)
    n_f5    = _soma_por_mes(ms[ident == 5], None, meses)

    serie = pd.DataFrame(index=meses)
    serie["A_todos_por_DT_INTER"] = a_todos
    serie["B_sem_ident5"] = b_sem5
    serie["B_ident5_por_DT_SAIDA"] = b5
    serie["B_total_zeros_0"] = serie["B_sem_ident5"] + serie["B_ident5_por_DT_SAIDA"]
    serie["B_total_zeros_1"] = serie["B_total_zeros_0"] + zeros
    serie["n_zeros"] = zeros
    serie["n_internacoes_sem_ident5"] = n_inter
    serie["n_saidas_sem_ident5"] = n_saida
    serie["n_registros_ident5"] = n_f5
    serie = serie.astype("int64")
    serie.index = [rotulo_mes(i) for i in meses]
    serie.index.name = "ano_mes"

    # ---- teste de identidade (IDENT != 5, registros dentro da janela) ----
    a0, bL = limites[0][0], limites[-1][1]
    dentro = (sc >= a0) & (ec <= bL)
    soma_datas = int((ec[dentro] - sc[dentro]).sum())
    dpc = dp[comum][dentro]
    soma_campo = int(dpc[dpc >= 0].sum())
    n_div = int((dpc != (ec[dentro] - sc[dentro])).sum())

    # ---- quanto a Abordagem A aloca fora da janela ou no mês errado ------
    fora = int(((mi[longa] < meses[0]) | (mi[longa] > meses[-1])).sum())
    dp_longa_mi_difere = int((mi[longa] != ms[longa]).sum())
    extra = (int(dentro.sum()), soma_datas, soma_campo, n_div, fora, dp_longa_mi_difere,
             int(longa.sum()))
    return serie, extra


def check_serie(d: dict) -> pd.DataFrame:
    titulo("CHECK 6 — Série mensal: Abordagem A x B, zeros e IDENT = 5")
    serie, (n_dentro, soma_datas, soma_campo, n_div, fora, mi_difere, n_longa) = construir_serie(d)

    log("Teste de identidade (IDENT != 5, registros inteiramente dentro da janela):")
    log(f"  registros                 : {n_dentro:,}")
    log(f"  soma dos dias pelas datas : {soma_datas:,}")
    log(f"  soma de DIAS_PERM         : {soma_campo:,}")
    log(f"  diferença                 : {soma_datas - soma_campo:,}")
    log(f"  registros com DIAS_PERM != diferença de datas: {n_div:,}")

    log("\nRegistros IDENT = 5 (fatias mensais):")
    log(f"  total                                        : {n_longa:,}")
    log(f"  com mês de DT_INTER != mês de DT_SAIDA       : {mi_difere:,} "
        f"({pct(mi_difere, n_longa)})")
    log(f"  cujo mês de DT_INTER cai FORA da janela      : {fora:,}")
    log("  (ou seja: na Abordagem A esses pacientes-dia iriam para o mês da")
    log("   internação original, não para o mês em que a ocupação ocorreu)")

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

    log("\nPrimeiros 15 meses da série (efeito de truncamento à esquerda):")
    log(serie[["A_todos_por_DT_INTER", "B_total_zeros_0", "B_ident5_por_DT_SAIDA"]].head(15).to_string())

    log("\nÚltimos 15 meses da série (efeito de censura à direita):")
    log(serie[["B_total_zeros_0", "B_ident5_por_DT_SAIDA", "n_saidas_sem_ident5",
               "n_registros_ident5"]].tail(15).to_string())

    return serie


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    somente6 = len(sys.argv) > 3 and sys.argv[3].strip() in {"6", "serie"}

    df = carregar()

    if not somente6:
        check_visao_geral(df)
        check_dias_perm_e_ident5(df)
        check_n_aih(df)
        check_defasagem(df)
        check_zeros(df)
        check_limites(df)
        gc.collect()

    dados = compactar_para_serie(df)
    del df
    gc.collect()
    serie = check_serie(dados)

    sufixo = "_check6" if somente6 else ""
    arq_rel = SAIDA / f"relatorio_diagnostico{sufixo}.txt"
    arq_csv = SAIDA / "serie_mensal_diagnostico.csv"
    serie.to_csv(arq_csv)
    arq_rel.write_text("\n".join(_linhas), encoding="utf-8")

    print("\n" + "-" * 74)
    print(f"Relatório : {arq_rel.resolve()}")
    print(f"Série CSV : {arq_csv.resolve()}")
    print("Envie esses dois arquivos.")


if __name__ == "__main__":
    main()
