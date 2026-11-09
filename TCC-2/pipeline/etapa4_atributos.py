#!/usr/bin/env python3
"""
etapa4_atributos.py
===================
Etapa 4 do pipeline: atributos de calendário, defasagens do alvo, proporções
defasadas e divisão cronológica em treinamento e teste.

Uso
---
    python pipeline/etapa4_atributos.py

Entrada (somente leitura):
    pipeline/saidas/etapa3_serie_mensal.parquet     128 meses, 01/2015 a 08/2025

Saídas (em pipeline/saidas/):
    etapa4_atributos.parquet    116 meses, 01/2016 a 08/2025, um por linha
    etapa4_relatorio.txt        contagens, conferências e resumo dos conjuntos

Colunas
-------
    mes, conjunto ("treino" ou "teste")
    y               pacientes-dia do mês (alvo em nível)
    y_diff          y - lag_1 (alvo da formulação em diferença)
    dias_no_mes, dez_jan, outono, inverno, primavera     calendário
    lag_1, lag_3, lag_12                                 y de 1, 3 e 12 meses antes
    prop_60mais_lag1, prop_masculino_lag1                proporções do mês anterior

Regras (ver CLAUDE.md)
----------------------
- Estações meteorológicas do hemisfério sul, por mês inteiro: verão = dez a fev
  (referência, sem coluna), outono = mar a mai, inverno = jun a ago, primavera =
  set a nov. dez_jan marca dezembro e janeiro, que têm a menor ocupação do verão.
- Nenhum atributo usa o próprio mês: defasagens e proporções vêm de meses anteriores.
  As proporções são ponderadas por pacientes-dia; a de idade exclui do denominador
  os pacientes-dia com idade inválida.
- Os 12 primeiros meses (2015) só servem para formar lag_12 e saem da tabela.
- Treino: 01/2016 a 08/2023 (92 meses). Teste: 09/2023 a 08/2025 (24 meses).
  Divisão cronológica, sem embaralhamento.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ    = Path(__file__).resolve().parent.parent
SAIDA   = RAIZ / "pipeline" / "saidas"
ENTRADA = SAIDA / "etapa3_serie_mensal.parquet"
ARQ_ATR = SAIDA / "etapa4_atributos.parquet"
ARQ_REL = SAIDA / "etapa4_relatorio.txt"

INICIO_TESTE = pd.Timestamp("2023-09-01")
DEFASAGENS   = [1, 3, 12]

CALENDARIO  = ["dias_no_mes", "dez_jan", "outono", "inverno", "primavera"]
LAGS        = [f"lag_{k}" for k in DEFASAGENS]
PROPORCOES  = ["prop_60mais_lag1", "prop_masculino_lag1"]

ESTACOES = {"outono": [3, 4, 5], "inverno": [6, 7, 8], "primavera": [9, 10, 11]}

# Valores de referência (CLAUDE.md, seção "Janela temporal" e Etapa 4)
REFERENCIA = {
    "meses lidos": 128,
    "meses com lag_12": 116,
    "meses de treino": 92,
    "meses de teste": 24,
    "primeiro mês de treino": "2016-01",
    "último mês de treino": "2023-08",
    "primeiro mês de teste": "2023-09",
    "último mês de teste": "2025-08",
    "valores nulos": 0,
    "meses não consecutivos": 0,
    "defasagens divergentes": 0,
    "y - lag_1 - y_diff diferente de zero": 0,
    "meses sem exatamente uma estação": 0,
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


# ============================================================
# ATRIBUTOS
# ============================================================

def montar(s: pd.DataFrame) -> pd.DataFrame:
    s = s.sort_values("mes").reset_index(drop=True)
    mes = s["mes"].dt.month
    d = pd.DataFrame({"mes": s["mes"], "y": s["pacientes_dia"]})

    d["dias_no_mes"] = s["mes"].dt.days_in_month.astype("int8")
    d["dez_jan"] = mes.isin([12, 1]).astype("int8")
    for nome, meses in ESTACOES.items():
        d[nome] = mes.isin(meses).astype("int8")

    for k in DEFASAGENS:
        d[f"lag_{k}"] = d["y"].shift(k)
    d["y_diff"] = d["y"] - d["lag_1"]

    d["prop_60mais_lag1"] = (s["pd_60mais"] / s["pd_idade_valida"]).shift(1)
    d["prop_masculino_lag1"] = (s["pd_masculino"] / s["pacientes_dia"]).shift(1)

    d = d.dropna(subset=LAGS).reset_index(drop=True)
    for nome in LAGS + ["y_diff"]:
        d[nome] = d[nome].astype("int64")
    d.insert(1, "conjunto", np.where(d["mes"] < INICIO_TESTE, "treino", "teste"))
    return d[["mes", "conjunto", "y", "y_diff"] + CALENDARIO + LAGS + PROPORCOES]


# ============================================================
# CONFERÊNCIA
# ============================================================

def conferir_defasagens(s: pd.DataFrame, d: pd.DataFrame) -> int:
    """Recalcula cada defasagem pela data, sem shift, e conta as divergências."""
    y = s.set_index("mes")["pacientes_dia"]
    prop60 = (s["pd_60mais"] / s["pd_idade_valida"]).set_axis(s["mes"])
    propm = (s["pd_masculino"] / s["pacientes_dia"]).set_axis(s["mes"])
    divergentes = 0
    for _, lin in d.iterrows():
        for k in DEFASAGENS:
            divergentes += int(y[lin["mes"] - pd.DateOffset(months=k)] != lin[f"lag_{k}"])
        anterior = lin["mes"] - pd.DateOffset(months=1)
        divergentes += int(not np.isclose(prop60[anterior], lin["prop_60mais_lag1"]))
        divergentes += int(not np.isclose(propm[anterior], lin["prop_masculino_lag1"]))
    return divergentes


def auditar(s: pd.DataFrame, d: pd.DataFrame) -> dict:
    tr, te = d[d["conjunto"] == "treino"], d[d["conjunto"] == "teste"]
    passos = d["mes"].dt.to_period("M").astype("int64").diff().dropna()
    estacoes = d[list(ESTACOES)].sum(axis=1)
    verao = d["mes"].dt.month.isin([12, 1, 2])
    return {
        "meses lidos": len(s),
        "meses com lag_12": len(d),
        "meses de treino": len(tr),
        "meses de teste": len(te),
        "primeiro mês de treino": f"{tr['mes'].min():%Y-%m}",
        "último mês de treino": f"{tr['mes'].max():%Y-%m}",
        "primeiro mês de teste": f"{te['mes'].min():%Y-%m}",
        "último mês de teste": f"{te['mes'].max():%Y-%m}",
        "valores nulos": int(d.isna().sum().sum()),
        "meses não consecutivos": int((passos != 1).sum()),
        "defasagens divergentes": conferir_defasagens(s, d),
        "y - lag_1 - y_diff diferente de zero": int((d["y"] - d["lag_1"] != d["y_diff"]).sum()),
        # cada mês fora do verão pertence a exatamente uma estação; no verão, a nenhuma
        "meses sem exatamente uma estação": int(((estacoes != 1) & ~verao).sum()
                                                + ((estacoes != 0) & verao).sum()),
    }


def indice_sazonal(s: pd.DataFrame) -> pd.Series:
    """Índice mensal (média 100) da ocupação média diária, só com meses anteriores ao
    teste: razão entre cada mês e a média móvel centrada de 12 meses (2x12)."""
    x = s[s["mes"] < INICIO_TESTE].set_index("mes")["pacientes_dia"]
    diaria = (x / x.index.days_in_month).to_numpy()
    pesos = np.r_[0.5, np.ones(11), 0.5] / 12
    media = np.full(len(diaria), np.nan)
    media[6:-6] = np.convolve(diaria, pesos, mode="valid")
    razao = pd.Series(diaria / media, index=x.index.month).dropna()
    indice = razao.groupby(level=0).mean()
    return indice / indice.mean() * 100


def relatar(s: pd.DataFrame, d: pd.DataFrame) -> None:
    tr, te = d[d["conjunto"] == "treino"], d[d["conjunto"] == "teste"]

    titulo("Conjuntos")
    for nome, x in [("treino", tr), ("teste", te)]:
        log(f"{nome:<7}: {x['mes'].min():%Y-%m} a {x['mes'].max():%Y-%m}, {len(x)} meses | "
            f"y de {x['y'].min():,} a {x['y'].max():,} (média {x['y'].mean():,.0f})")

    acima = te[te["y"] > tr["y"].max()]
    log(f"\nMeses de teste com y acima do maior valor do treino ({tr['y'].max():,}): "
        f"{len(acima)} de {len(te)}")
    log(f"Meses de teste com y_diff fora da faixa do treino "
        f"({tr['y_diff'].min():,} a {tr['y_diff'].max():,}): "
        f"{int(((te['y_diff'] < tr['y_diff'].min()) | (te['y_diff'] > tr['y_diff'].max())).sum())} "
        f"de {len(te)}")

    titulo("Atributos")
    log(f"Calendário : {', '.join(CALENDARIO)}")
    log(f"Defasagens : {', '.join(LAGS)}")
    log(f"Proporções : {', '.join(PROPORCOES)} (só entram na comparação com o calendário)")
    log("\nContagem de meses por atributo binário no treino:")
    for nome in ["dez_jan"] + list(ESTACOES):
        log(f"  {nome:<10}: {int(tr[nome].sum())}")
    log("\nCorrelação de cada atributo com y no treino:")
    for nome in CALENDARIO + LAGS + PROPORCOES:
        log(f"  {nome:<20}: {tr[nome].corr(tr['y']):+.3f}")
    log("\nResumo das proporções no treino:")
    log(tr[PROPORCOES].describe().round(4).to_string())

    ind = indice_sazonal(s)
    log("\nÍndice sazonal da ocupação média diária (01/2015 a 08/2023, média = 100):")
    log("  " + "  ".join(f"{m:>2}: {v:5.1f}" for m, v in ind.items()))
    verao = ind[[12, 1, 2]]
    log(f"  verão (dez a fev): de {verao.min():.1f} a {verao.max():.1f}, "
        f"amplitude {verao.max() - verao.min():.1f}")
    for nome, meses in ESTACOES.items():
        log(f"  {nome:<9}: média {ind[meses].mean():.1f}")


def conferir(obtido: dict) -> bool:
    titulo("CONFERÊNCIA COM OS VALORES DE REFERÊNCIA")
    log(f"{'item':<40}{'esperado':>12}{'obtido':>12}  situação")
    tudo_ok = True
    for item, esperado in REFERENCIA.items():
        ok = obtido[item] == esperado
        tudo_ok &= ok
        log(f"{item:<40}{str(esperado):>12}{str(obtido[item]):>12}  {'OK' if ok else 'DIVERGE'}")
    return tudo_ok


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not ENTRADA.is_file():
        sys.exit(f"Arquivo de entrada não encontrado: {ENTRADA}. Rodar a Etapa 3 antes.")

    titulo("ETAPA 4 — Atributos e divisão em treinamento e teste")
    log(f"Entrada: {ENTRADA.relative_to(RAIZ)}")

    s = pd.read_parquet(ENTRADA)
    d = montar(s)
    relatar(s, d)
    tudo_ok = conferir(auditar(s, d))

    if tudo_ok:
        temp = ARQ_ATR.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pandas(d, preserve_index=False), temp)
        temp.replace(ARQ_ATR)
        log(f"\nAtributos gravados: {ARQ_ATR.relative_to(RAIZ)} ({len(d)} meses, "
            f"{d.shape[1]} colunas)")
    else:
        log("\nCONFERÊNCIA FALHOU: atributos NÃO gravados. Investigar os itens 'DIVERGE'.")

    ARQ_REL.write_text("\n".join(_linhas), encoding="utf-8")
    print(f"Relatório: {ARQ_REL.relative_to(RAIZ)}")
    sys.exit(0 if tudo_ok else 1)


if __name__ == "__main__":
    main()
