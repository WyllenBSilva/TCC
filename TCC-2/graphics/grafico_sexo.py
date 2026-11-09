#!/usr/bin/env python3
"""
grafico_sexo.py
===============
Gráfico de pizza da proporção de registros por sexo, na base inteira.

Uso
---
    python graphics/grafico_sexo.py

Entrada (somente leitura):
    pipeline/saidas/etapa1_base_limpa.parquet

Saídas:
    monografia/figuras/distribuicao_sexo.png
    graphics/saidas/distribuicao_sexo.csv

O código 0 (ignorado) só aparece no gráfico se ocorrer na base. Um código fora de
SEXOS interrompe o script. A coluna dias_perm do CSV soma DIAS_PERM de cada grupo,
para comparar a proporção por registro com a proporção por dia de permanência.
"""

from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from comum import AZUL, LARANJA, GRADE, TABELAS, aplicar_estilo, lotes, percentual, salvar, total_registros


# código: (rótulo, cor, hachura para impressão em escala de cinza)
SEXOS = {
    "1": ("Masculino", AZUL, "///"),
    "3": ("Feminino", LARANJA, "\\\\\\"),
    "0": ("Ignorado", GRADE, ""),
}


def contar() -> pd.DataFrame:
    registros, dias = Counter(), Counter()
    for d in lotes(["SEXO", "DIAS_PERM"]):
        grupos = d.groupby("SEXO", observed=True)["DIAS_PERM"].agg(["size", "sum"])
        for codigo, linha in grupos.iterrows():
            registros[codigo] += int(linha["size"])
            dias[codigo] += int(linha["sum"])

    desconhecidos = set(registros) - set(SEXOS)
    if desconhecidos:
        raise SystemExit(f"Código de SEXO não previsto: {sorted(desconhecidos)}")
    if sum(registros.values()) != total_registros():
        raise SystemExit("A soma dos registros por sexo difere do total da base.")

    t = pd.DataFrame([
        {"codigo": c, "sexo": SEXOS[c][0], "registros": registros[c], "dias_perm": dias[c]}
        for c in SEXOS
    ])
    t["pct_registros"] = 100 * t["registros"] / t["registros"].sum()
    t["pct_dias_perm"] = 100 * t["dias_perm"] / t["dias_perm"].sum()
    return t


def desenhar(t: pd.DataFrame) -> None:
    aplicar_estilo()
    presentes = t[t["registros"] > 0]

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    fatias, _ = ax.pie(
        presentes["registros"],
        labels=[percentual(p) for p in presentes["pct_registros"]],
        colors=[SEXOS[c][1] for c in presentes["codigo"]],
        hatch=[SEXOS[c][2] for c in presentes["codigo"]],
        startangle=90,
        counterclock=False,
        labeldistance=1.08,
        textprops={"fontsize": 13},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax.legend(
        fatias,
        presentes["sexo"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncols=len(presentes),
        frameon=False,
    )
    ax.set_aspect("equal")
    salvar(fig, "distribuicao_sexo.png")


def main() -> None:
    t = contar()
    TABELAS.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABELAS / "distribuicao_sexo.csv", index=False)
    print(t.to_string(index=False))
    desenhar(t)


if __name__ == "__main__":
    main()
