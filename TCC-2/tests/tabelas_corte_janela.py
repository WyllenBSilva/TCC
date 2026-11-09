#!/usr/bin/env python3
"""
tabelas_corte_janela.py
=======================
Gera as tabelas que justificam o corte da série de modelagem em 08/2025: queda dos
meses finais de 2025 em relação aos anos anteriores e déficit esperado pela
simulação de faturamento.

Uso
---
    python tests/tabelas_corte_janela.py

Entradas (somente leitura):
    pipeline/saidas/etapa3_serie_completa.parquet    série final, 01/2015 a 12/2025
    tests/saidas/completude_por_mes_e_defasagem.csv  saída de valida_corte_janela.py

Saídas (em tests/saidas/):
    corte_mesmo_mes_ano_a_ano.csv   pacientes-dia de 2023 a 2025 por mês do ano,
                                    variação 2025/2024, mediana da variação 2016-2024
                                    e déficit médio simulado
    corte_total_anual.csv           total por ano e total de janeiro a agosto
    relatorio_tabelas_corte.txt     as mesmas tabelas, formatadas

Déficit simulado: com dados até a competência 12/2025, o mês M de 2025 tem k = 12 - M
competências posteriores. O déficit de cada k é a média, nos meses de 2023-01 a
2024-12, de 1 - (pacientes-dia conhecidos com k competências posteriores) / (total
final do mês). Só vale para k <= 6, o maior k da simulação.
"""

from pathlib import Path

import numpy as np
import pandas as pd


RAIZ    = Path(__file__).resolve().parent.parent
SERIE   = RAIZ / "pipeline" / "saidas" / "etapa3_serie_completa.parquet"
COMPLET = RAIZ / "tests" / "saidas" / "completude_por_mes_e_defasagem.csv"
SAIDA   = RAIZ / "tests" / "saidas"

MESES = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho", "Agosto",
         "Setembro", "Outubro", "Novembro", "Dezembro"]

_linhas: list[str] = []


def log(texto="") -> None:
    print(texto)
    _linhas.append(str(texto))


def deficit_simulado() -> pd.Series:
    """Déficit médio (%) por número k de competências posteriores."""
    c = pd.read_csv(COMPLET, index_col="ano_mes")
    ref = c.loc["2023-01":"2024-12"]
    return pd.Series({k: 100 * (1 - ref[f"ate_k{k}"] / ref["pacientes_dia"]).mean()
                      for k in range(7)})


def tabela_mensal(d: pd.DataFrame, deficit: pd.Series) -> pd.DataFrame:
    p = d.pivot(index="m", columns="ano", values="pacientes_dia")
    var = (p / p.shift(axis=1) - 1) * 100
    t = pd.DataFrame({
        "mes": MESES,
        "pd_2023": p[2023].to_numpy(),
        "pd_2024": p[2024].to_numpy(),
        "pd_2025": p[2025].to_numpy(),
        "var_2025_2024_pct": var[2025].round(1).to_numpy(),
        "mediana_var_2016_2024_pct": var.loc[:, 2016:2024].median(axis=1).round(1).to_numpy(),
    })
    k = 12 - np.arange(1, 13)
    t["k_competencias_posteriores"] = k
    t["deficit_simulado_pct"] = [round(deficit[x], 2) if x in deficit.index else np.nan for x in k]
    return t


def tabela_anual(d: pd.DataFrame) -> pd.DataFrame:
    tot = d.groupby("ano")["pacientes_dia"].sum()
    jan_ago = d[d["m"] <= 8].groupby("ano")["pacientes_dia"].sum()
    return pd.DataFrame({
        "ano": tot.index,
        "total": tot.to_numpy(),
        "var_total_pct": (tot.pct_change() * 100).round(1).to_numpy(),
        "total_jan_ago": jan_ago.to_numpy(),
        "var_jan_ago_pct": (jan_ago.pct_change() * 100).round(1).to_numpy(),
    })


def main() -> None:
    d = pd.read_parquet(SERIE)
    d["ano"], d["m"] = d["mes"].dt.year, d["mes"].dt.month

    mensal = tabela_mensal(d, deficit_simulado())
    anual = tabela_anual(d)
    mensal.to_csv(SAIDA / "corte_mesmo_mes_ano_a_ano.csv", index=False)
    anual.to_csv(SAIDA / "corte_total_anual.csv", index=False)

    log("Pacientes-dia por mês do ano (série final, com reconstrução das lacunas)")
    log("k = competências posteriores disponíveis para o mês de 2025 (base até 12/2025)\n")
    log(mensal.to_string(index=False))
    log("\nTotal por ano (2025 inclui os meses de setembro a dezembro incompletos)\n")
    log(anual.to_string(index=False))

    a = anual.set_index("ano")
    log(f"\nJaneiro a agosto de 2025 contra 2024: {a.at[2025, 'var_jan_ago_pct']:+.1f}% "
        f"({a.at[2025, 'total_jan_ago']:,} pacientes-dia)")
    log(f"Faixa da mediana 2016-2024 da variação anual de cada mês: "
        f"{mensal['mediana_var_2016_2024_pct'].min():+.1f}% a "
        f"{mensal['mediana_var_2016_2024_pct'].max():+.1f}%")

    (SAIDA / "relatorio_tabelas_corte.txt").write_text("\n".join(_linhas), encoding="utf-8")


if __name__ == "__main__":
    main()
