#!/usr/bin/env python3
"""
diagnostico_complementar.py
===========================
Dois pontos que ficaram abertos no CHECK 6:

  CHECK 7  O pico da Abordagem A em julho/2016 (454.566 pacientes-dia,
           contra ~195.000 nos meses vizinhos). A hipótese é que exista
           uma coorte de pacientes de longa permanência com DT_INTER
           naquele mês, cujas fatias mensais a Abordagem A devolve todas
           para o mês da internação original.

  CHECK 8  A subnotificação de fatias IDENT = 5 nas competências 2016-07
           (196 registros, contra ~350) e 2021-11 (93, contra ~180).
           Mede quantos pacientes têm fatia no mês anterior E no mês
           seguinte, mas não no mês em questão — isto é, estavam
           internados e a parcela não foi apresentada — e quantos
           pacientes-dia isso representa.

Uso
---
    python diagnostico_complementar.py [entrada] [pasta_saida]

    entrada = pasta com .parquet ou um único .parquet
              (padrão: data_all_concatenado)

Saída
-----
    diagnostico_metodologico/relatorio_complementar.txt
    diagnostico_metodologico/lacunas_longa_permanencia.csv

Só lê os dados; não altera nenhum arquivo de entrada.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ENTRADA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data_all_concatenado")
SAIDA   = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("diagnostico_metodologico")

COLUNAS = ["N_AIH", "DT_INTER", "DT_SAIDA", "DIAS_PERM", "IDENT"]
ANO_INI, ANO_FIM = 2015, 2025
TAM_LOTE = 500_000

_linhas: list[str] = []


def log(texto="") -> None:
    print(texto)
    _linhas.append(str(texto))


def titulo(texto: str) -> None:
    log("\n" + "=" * 74)
    log(texto)
    log("=" * 74)


def rotulo(idx) -> str:
    return f"{int(idx) // 12}-{int(idx) % 12 + 1:02d}"


def _data(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        txt = s.astype("Int64").astype("string")
    else:
        txt = s.astype("string").str.strip()
    return pd.to_datetime(txt, format="%Y%m%d", errors="coerce")


def carregar() -> pd.DataFrame:
    arquivos = [ENTRADA] if ENTRADA.is_file() else sorted(ENTRADA.glob("*.parquet"))
    if not arquivos:
        sys.exit(f"Nenhum .parquet encontrado em: {ENTRADA.resolve()}")
    print(f"Lendo {len(arquivos)} arquivo(s) de {ENTRADA} ...")

    partes, total = [], 0
    for arquivo in arquivos:
        pf = pq.ParquetFile(arquivo)
        cols = [c for c in COLUNAS if c in pf.schema_arrow.names]
        for lote in pf.iter_batches(batch_size=TAM_LOTE, columns=cols):
            p = lote.to_pandas()
            ident = pd.to_numeric(p["IDENT"], errors="coerce").fillna(-1).astype("int8")
            # só as fatias de longa permanência interessam aqui
            p = p[ident == 5]
            if p.empty:
                continue
            di, ds = _data(p["DT_INTER"]), _data(p["DT_SAIDA"])
            partes.append(pd.DataFrame({
                "N_AIH": pd.to_numeric(p["N_AIH"], errors="coerce").fillna(0).astype("int64"),
                "DP": pd.to_numeric(p["DIAS_PERM"], errors="coerce").fillna(0).astype("int32"),
                "MI": (di.dt.year * 12 + di.dt.month - 1).fillna(-1).astype("int32"),
                "MS": (ds.dt.year * 12 + ds.dt.month - 1).fillna(-1).astype("int32"),
            }))
            total += len(p)
            print(f"  ... {total:,} fatias IDENT=5 lidas", end="\r", flush=True)
    print()
    if not partes:
        sys.exit("Nenhum registro IDENT = 5 encontrado.")
    return pd.concat(partes, ignore_index=True)


# ============================================================
# CHECK 7 — o pico da Abordagem A
# ============================================================

def check_pico_abordagem_a(f: pd.DataFrame) -> None:
    titulo("CHECK 7 — Fatias IDENT = 5 por MÊS DA INTERNAÇÃO (origem do pico da Abordagem A)")

    g = f.groupby("MI").agg(fatias=("DP", "size"), pac_dia=("DP", "sum"),
                            pacientes=("N_AIH", "nunique"))
    g = g.sort_values("pac_dia", ascending=False)
    log("Dez meses de internação que concentram mais pacientes-dia de longa permanência.")
    log("Na Abordagem A, TODO esse volume é lançado no mês da internação:\n")
    log(f"  {'mês DT_INTER':<14}{'pacientes':>11}{'fatias':>9}{'pacientes-dia':>16}"
        f"{'dias/paciente':>15}")
    for mi, r in g.head(10).iterrows():
        log(f"  {rotulo(mi):<14}{int(r.pacientes):>11,}{int(r.fatias):>9,}"
            f"{int(r.pac_dia):>16,}{r.pac_dia / r.pacientes:>15,.0f}")

    log(f"\nTotal de pacientes-dia em fatias IDENT = 5: {int(f['DP'].sum()):,}")
    top = g.head(10)["pac_dia"].sum()
    log(f"Os dez meses acima concentram {top:,.0f} "
        f"({top / f['DP'].sum():.1%}) desse total.")

    alvo = 2016 * 12 + 6  # julho/2016
    if alvo in g.index:
        r = g.loc[alvo]
        log(f"\nJulho/2016 em detalhe: {int(r.pacientes):,} pacientes de longa permanência "
            f"internados nesse mês,")
        log(f"  somando {int(r.pac_dia):,} pacientes-dia distribuídos em {int(r.fatias):,} "
            f"fatias mensais")
        log(f"  ({r.pac_dia / r.pacientes:,.0f} dias por paciente, ou "
            f"{r.pac_dia / r.pacientes / 365:.1f} anos em média).")
        sub = f[f["MI"] == alvo]
        log(f"  Meses de ocupação cobertos: {rotulo(sub['MS'].min())} a {rotulo(sub['MS'].max())}.")


# ============================================================
# CHECK 8 — lacunas nas cadeias de longa permanência
# ============================================================

def check_lacunas(f: pd.DataFrame) -> pd.DataFrame:
    titulo("CHECK 8 — Lacunas nas cadeias: paciente internado cuja fatia não foi apresentada")

    f = f[(f["MS"] >= ANO_INI * 12) & (f["MS"] <= ANO_FIM * 12 + 11)]
    presente = set(zip(f["N_AIH"].to_numpy(), f["MS"].to_numpy()))

    # para cada cadeia, os meses entre a primeira e a última fatia
    lim = f.groupby("N_AIH")["MS"].agg(["min", "max"])
    lim = lim[lim["max"] > lim["min"]]
    log(f"Cadeias com duas ou mais fatias: {len(lim):,}")

    faltas = []
    for aih, (m0, m1) in zip(lim.index, lim.to_numpy()):
        for m in range(m0 + 1, m1):
            if (aih, m) not in presente:
                faltas.append((aih, m))
    log(f"Lacunas internas (mês entre a primeira e a última fatia, sem registro): {len(faltas):,}")

    if not faltas:
        log("Nenhuma lacuna: todas as cadeias são contínuas.")
        return pd.DataFrame()

    lac = pd.DataFrame(faltas, columns=["N_AIH", "MS"])
    lac["dias_no_mes"] = [pd.Period(rotulo(m), freq="M").days_in_month for m in lac["MS"]]

    por_mes = lac.groupby("MS").agg(pacientes=("N_AIH", "size"),
                                    pac_dia_faltantes=("dias_no_mes", "sum"))
    obs = f.groupby("MS")["DP"].agg(fatias="size", pac_dia="sum")
    tab = por_mes.join(obs, how="left").fillna(0)
    tab["pac_dia_corrigido"] = tab["pac_dia"] + tab["pac_dia_faltantes"]
    tab["subnotificacao_%"] = (tab["pac_dia_faltantes"] / tab["pac_dia_corrigido"] * 100).round(1)
    tab.index = [rotulo(m) for m in tab.index]
    tab.index.name = "ano_mes"

    log(f"\nMeses de ocupação afetados: {len(tab)}")
    log("\nDez meses com maior subnotificação de pacientes-dia de longa permanência:")
    log(tab.sort_values("pac_dia_faltantes", ascending=False).head(10).to_string())

    for alvo in ["2016-07", "2021-11"]:
        if alvo in tab.index:
            r = tab.loc[alvo]
            log(f"\n{alvo}: {int(r.pacientes):,} pacientes sem fatia apresentada, "
                f"{int(r.pac_dia_faltantes):,} pacientes-dia")
            log(f"  observado {int(r.pac_dia):,} -> reconstruído {int(r.pac_dia_corrigido):,} "
                f"({r['subnotificacao_%']:.1f}% faltante)")

    total = int(lac["dias_no_mes"].sum())
    log(f"\nTotal reconstruível: {total:,} pacientes-dia "
        f"({total / f['DP'].sum():.2%} do volume de longa permanência)")
    log("Cada lacuna é um paciente com fatia antes E depois do mês ausente, "
        "ou seja,\ncomprovadamente internado — o dia-leito existiu e não foi faturado "
        "naquela competência.")
    return tab


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    f = carregar()
    log(f"Fatias IDENT = 5 carregadas: {len(f):,}")
    check_pico_abordagem_a(f)
    tab = check_lacunas(f)

    rel = SAIDA / "relatorio_complementar.txt"
    rel.write_text("\n".join(_linhas), encoding="utf-8")
    csv = SAIDA / "lacunas_longa_permanencia.csv"
    if len(tab):
        tab.to_csv(csv)

    print("\n" + "-" * 74)
    print(f"Relatório : {rel.resolve()}")
    if len(tab):
        print(f"Lacunas   : {csv.resolve()}")


if __name__ == "__main__":
    main()