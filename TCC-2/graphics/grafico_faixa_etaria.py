#!/usr/bin/env python3
"""
grafico_faixa_etaria.py
=======================
Gráficos de barras por faixa etária, na base inteira: um com a contagem de registros
e outro com a soma de DIAS_PERM.

Uso
---
    python graphics/grafico_faixa_etaria.py

Entrada (somente leitura):
    pipeline/saidas/etapa1_base_limpa.parquet

Saídas:
    monografia/figuras/distribuicao_faixa_etaria.png
    monografia/figuras/distribuicao_faixa_etaria_dias.png
    graphics/saidas/distribuicao_faixa_etaria.csv

As faixas têm larguras diferentes (1 a 4 anos cobre quatro idades; 20 a 29, dez), de
modo que a altura das barras compara o volume de cada faixa, não a densidade por ano
de idade. Registros com IDADE_ANOS nula (idade inválida) ficam fora das barras e
aparecem no CSV na linha "Idade inválida".
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comum import AZUL, GRADE, TABELAS, aplicar_estilo, lotes, milhar, salvar, total_registros


# Limite inferior de cada faixa a partir da segunda; IDADE_ANOS = 0 é "menos de 1 ano".
LIMITES = [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80]
ROTULOS = ["<1", "1–4", "5–9", "10–14", "15–19", "20–29", "30–39",
           "40–49", "50–59", "60–69", "70–79", "80+"]
INVALIDA = "Idade inválida"


def contar() -> pd.DataFrame:
    n = len(ROTULOS) + 1
    registros, dias = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    for d in lotes(["IDADE_ANOS", "DIAS_PERM"]):
        idade = d["IDADE_ANOS"].to_numpy(dtype=float)
        faixa = np.searchsorted(LIMITES, idade, side="right")
        faixa[np.isnan(idade)] = n - 1
        np.add.at(registros, faixa, 1)
        np.add.at(dias, faixa, d["DIAS_PERM"].to_numpy(dtype=np.int64))

    if registros.sum() != total_registros():
        raise SystemExit("A soma dos registros por faixa difere do total da base.")

    t = pd.DataFrame({"faixa": ROTULOS + [INVALIDA], "registros": registros, "dias_perm": dias})
    validos = t["faixa"] != INVALIDA
    t["pct_registros"] = (100 * t["registros"] / t.loc[validos, "registros"].sum()).where(validos)
    t["pct_dias_perm"] = (100 * t["dias_perm"] / t.loc[validos, "dias_perm"].sum()).where(validos)
    return t


def desenhar(t: pd.DataFrame, coluna: str, rotulo: str, nome: str) -> None:
    aplicar_estilo()
    barras = t[t["faixa"] != INVALIDA]
    x = np.arange(len(barras))

    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    ax.bar(x, barras[coluna], width=0.8, color=AZUL, zorder=2)

    ax.set_xticks(x, barras["faixa"], fontsize=8)
    ax.set_xlabel("Faixa etária (anos)")
    ax.set_ylabel(rotulo)
    ax.yaxis.set_major_formatter(lambda v, _: milhar(v))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRADE, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)

    pico = barras[coluna].to_numpy().argmax()
    ax.annotate(milhar(barras[coluna].iloc[pico]), (pico, barras[coluna].iloc[pico]),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)

    salvar(fig, nome)


def main() -> None:
    t = contar()
    TABELAS.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABELAS / "distribuicao_faixa_etaria.csv", index=False)
    print(t.to_string(index=False))
    desenhar(t, "registros", "Registros", "distribuicao_faixa_etaria.png")
    desenhar(t, "dias_perm", "Dias de permanência", "distribuicao_faixa_etaria_dias.png")


if __name__ == "__main__":
    main()
