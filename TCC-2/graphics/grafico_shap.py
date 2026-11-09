#!/usr/bin/env python3
"""
grafico_shap.py
===============
Gráficos dos valores SHAP calculados na Etapa 6, nos 24 meses de teste.

Uso
---
    python graphics/grafico_shap.py

Entradas (somente leitura):
    pipeline/saidas/etapa6_shap_importancia.csv
    pipeline/saidas/etapa6_shap_valores.parquet

Saídas:
    monografia/figuras/shap_importancia.png   média de |SHAP| por atributo, com o
                                              desvio-padrão entre as 10 sementes
    monografia/figuras/shap_dispersao.png     SHAP de cada mês de teste, colorido
                                              pelo valor do atributo

Linhas: formulação (nível em cima, diferença embaixo); colunas: algoritmo. Em cada
linha o eixo x é compartilhado, para comparar os dois algoritmos na mesma escala. Em
diferença, os valores SHAP explicam a variação mensal prevista (y - lag_1).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from comum import AZUL, CINZA, LARANJA, PIPELINE, TINTA, aplicar_estilo, eixo_limpo, milhar, salvar


LINHAS  = {"nivel": "em nível", "diferenca": "em diferença"}
COLUNAS = {"random_forest": ("Random Forest", AZUL), "mlp": ("MLP", LARANJA)}
MAPA = LinearSegmentedColormap.from_list("baixo_alto", [AZUL, "#c8c8c8", LARANJA])


def painel(eixos):
    """Percorre os painéis 2 x 2 com (eixo, algoritmo, formulação)."""
    for i, form in enumerate(LINHAS):
        for j, alg in enumerate(COLUNAS):
            yield eixos[i, j], alg, form


def titulo_painel(ax, alg: str, form: str) -> None:
    ax.set_title(f"{COLUNAS[alg][0]} {LINHAS[form]}", fontsize=10, loc="left")


def ordem_atributos(imp: pd.DataFrame, alg: str, form: str) -> list[str]:
    """Atributos do mais para o menos importante no teste, pelo Random Forest da mesma
    formulação, para que os dois painéis de uma linha tenham a mesma ordem."""
    t = imp[(imp["configuracao"] == f"random_forest_{form}") & (imp["conjunto"] == "teste")]
    return t.sort_values("media_abs_shap", ascending=False)["atributo"].tolist()


def grafico_importancia(imp: pd.DataFrame) -> None:
    aplicar_estilo()
    fig, eixos = plt.subplots(2, 2, figsize=(6.3, 5.6), sharex="row",
                              gridspec_kw={"height_ratios": [8, 5]})
    for ax, alg, form in painel(eixos):
        ordem = ordem_atributos(imp, alg, form)
        t = imp[(imp["configuracao"] == f"{alg}_{form}") & (imp["conjunto"] == "teste")]
        t = t.set_index("atributo").loc[ordem]
        y = np.arange(len(t))
        ax.barh(y, t["media_abs_shap"], height=0.65, color=COLUNAS[alg][1], zorder=2)
        ax.errorbar(t["media_abs_shap"], y, xerr=t["dp_entre_sementes"], fmt="none",
                    ecolor=TINTA, elinewidth=1, capsize=2, zorder=3)
        ax.set_yticks(y, ordem, fontsize=8)
        ax.invert_yaxis()
        linha = imp[imp["configuracao"].str.endswith(form) & (imp["conjunto"] == "teste")]
        ax.set_xlim(0, 1.05 * (linha["media_abs_shap"] + linha["dp_entre_sementes"]).max())
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(lambda v, _: milhar(v))
        ax.tick_params(axis="x", labelsize=8)
        titulo_painel(ax, alg, form)
        eixo_limpo(ax, grade="x")
    for ax in eixos[1]:
        ax.set_xlabel("Média de |SHAP| no teste (pacientes-dia)", fontsize=9)
    fig.tight_layout()
    salvar(fig, "shap_importancia.png")


def grafico_dispersao(imp: pd.DataFrame, val: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    aplicar_estilo()
    fig, eixos = plt.subplots(2, 2, figsize=(6.3, 5.8), sharex="row",
                              gridspec_kw={"height_ratios": [8, 5]})
    for ax, alg, form in painel(eixos):
        ordem = ordem_atributos(imp, alg, form)
        v = val[(val["configuracao"] == f"{alg}_{form}") & (val["conjunto"] == "teste")]
        for i, atributo in enumerate(ordem):
            a = v[v["atributo"] == atributo]
            x = a["valor_atributo"].to_numpy()
            cor = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.full_like(x, 0.5)
            ax.scatter(a["shap"], i + rng.uniform(-0.22, 0.22, len(a)), c=cor, cmap=MAPA,
                       vmin=0, vmax=1, s=14, edgecolors="white", linewidths=0.4, zorder=3)
        ax.axvline(0, color=CINZA, linewidth=0.8, zorder=1)
        ax.set_yticks(range(len(ordem)), ordem, fontsize=8)
        ax.set_ylim(len(ordem) - 0.5, -0.5)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(lambda v, _: milhar(v))
        ax.tick_params(axis="x", labelsize=8)
        titulo_painel(ax, alg, form)
        eixo_limpo(ax, grade="x")
    for ax in eixos[1]:
        ax.set_xlabel("Valor SHAP (pacientes-dia)", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    barra = fig.add_axes((0.3, 0.025, 0.4, 0.018))
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=MAPA), cax=barra, orientation="horizontal",
                      ticks=[0, 1])
    cb.ax.set_xticklabels(["baixo", "alto"], fontsize=8)
    cb.outline.set_visible(False)
    cb.set_label("Valor do atributo", fontsize=8, labelpad=-8)
    salvar(fig, "shap_dispersao.png")


def main() -> None:
    imp = pd.read_csv(PIPELINE / "etapa6_shap_importancia.csv")
    val = pd.read_parquet(PIPELINE / "etapa6_shap_valores.parquet")
    grafico_importancia(imp)
    grafico_dispersao(imp, val)


if __name__ == "__main__":
    main()
