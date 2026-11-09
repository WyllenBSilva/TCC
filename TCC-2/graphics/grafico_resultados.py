#!/usr/bin/env python3
"""
grafico_resultados.py
=====================
Gráficos da divisão dos dados e das previsões dos modelos selecionados na Etapa 5.

Uso
---
    python graphics/grafico_resultados.py

Entradas (somente leitura):
    pipeline/saidas/etapa3_serie_mensal.parquet   série mensal, 01/2015 a 08/2025
    pipeline/saidas/etapa4_atributos.parquet      divisão treino/teste
    pipeline/saidas/etapa5_previsoes.parquet      previsão média entre sementes
    pipeline/saidas/etapa5_resultados.csv         configurações selecionadas

Saídas:
    monografia/figuras/serie_mensal_divisao.png   série inteira com treino e teste
    monografia/figuras/previsoes_teste.png        observado x previsto nos 24 meses
                                                  de teste, um painel por formulação
    monografia/figuras/dispersao_teste.png        previsto contra observado no teste
    graphics/saidas/previsoes_teste.csv           os valores das duas últimas figuras
    graphics/saidas/erros_teste.csv               resumo dos erros da previsão média no
                                                  teste, usado no texto dos resultados

As previsões são a média das 10 sementes de cada configuração selecionada (o conjunto
de atributos escolhido pela validação). A linha "maior valor do treino" mostra o teto
que uma floresta em nível não consegue ultrapassar.

O MAE de erros_teste.csv é o da previsão média entre sementes (o que as figuras
mostram), não a média dos MAE de cada semente de etapa5_resultados.csv. Os dois quase
coincidem quando as sementes concordam; diferem quando elas variam muito (MLP em nível).
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comum import (AZUL, CINZA, FUNDO, LARANJA, PIPELINE, TABELAS, TINTA, aplicar_estilo,
                   eixo_limpo, milhar, salvar)


FORMULACOES = {"nivel": "Formulação em nível", "diferenca": "Formulação em diferença"}
MODELOS = {
    "random_forest": {"rotulo": "Random Forest", "cor": AZUL, "linha": "--", "marcador": "o"},
    "mlp":           {"rotulo": "MLP",           "cor": LARANJA, "linha": "-.", "marcador": "^"},
}
GRADE_HACHURA = "#c4c4c4"


def colunas_selecionadas() -> dict:
    """(modelo, formulação) -> coluna de etapa5_previsoes com o conjunto selecionado."""
    r = pd.read_csv(PIPELINE / "etapa5_resultados.csv")
    r = r[r["selecionado"] & ~r["modelo"].str.startswith("ingenua")]
    return {(x.modelo, x.formulacao): f"{x.modelo}_{x.formulacao}_{x.atributos}"
            for x in r.itertuples()}


def eixo_meses(ax, intervalo: int) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=intervalo))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.yaxis.set_major_formatter(lambda v, _: milhar(v))


def grafico_serie() -> None:
    s = pd.read_parquet(PIPELINE / "etapa3_serie_mensal.parquet")
    a = pd.read_parquet(PIPELINE / "etapa4_atributos.parquet")
    tr, te = a[a["conjunto"] == "treino"]["mes"], a[a["conjunto"] == "teste"]["mes"]
    meio_mes = pd.Timedelta(days=15)

    aplicar_estilo()
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    regioes = [
        (s["mes"].min(), tr.min(), "2015\n(só defasagens)", "////"),
        (tr.min(), te.min(), f"Treino ({len(tr)} meses)", None),
        (te.min(), te.max() + 2 * meio_mes, f"Teste ({len(te)} meses)", None),
    ]
    for ini, fim, texto, hachura in regioes:
        ini, fim = ini - meio_mes, fim - meio_mes
        if hachura or "Teste" in texto:
            ax.axvspan(ini, fim, facecolor=FUNDO, edgecolor=GRADE_HACHURA, hatch=hachura,
                       linewidth=0, zorder=0)
        ax.text(ini + (fim - ini) / 2, 1.01, texto, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, color=TINTA)

    ax.plot(s["mes"], s["pacientes_dia"], color=TINTA, linewidth=1.6, zorder=3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(lambda v, _: milhar(v))
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylabel("Pacientes-dia")
    ax.set_xlim(s["mes"].min() - meio_mes, s["mes"].max() + meio_mes)
    eixo_limpo(ax)
    salvar(fig, "serie_mensal_divisao.png")


def grafico_previsoes(te: pd.DataFrame, cols: dict, teto: float) -> None:
    aplicar_estilo()
    fig, eixos = plt.subplots(2, 1, figsize=(6.3, 6.4), sharex=True, sharey=True)
    for ax, (form, titulo) in zip(eixos, FORMULACOES.items()):
        ax.plot(te["mes"], te["y"], color=TINTA, linewidth=1.8, marker="s", markersize=4,
                label="Observado", zorder=4)
        for alg, est in MODELOS.items():
            ax.plot(te["mes"], te[cols[(alg, form)]], color=est["cor"], linestyle=est["linha"],
                    linewidth=1.8, marker=est["marcador"], markersize=4, label=est["rotulo"],
                    zorder=3)
        ax.axhline(teto, color=CINZA, linestyle=":", linewidth=1.2, zorder=2)
        ax.annotate("Maior valor do treino", (te["mes"].iloc[0], teto), xytext=(0, 4),
                    textcoords="offset points", fontsize=8, color=CINZA, va="bottom")
        ax.set_title(titulo, fontsize=11, loc="left")
        ax.set_ylabel("Pacientes-dia")
        eixo_meses(ax, 3)
        eixo_limpo(ax)
    eixos[-1].tick_params(axis="x", labelsize=8)
    fig.legend(*eixos[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.11, hspace=0.3)
    salvar(fig, "previsoes_teste.png")


def grafico_dispersao(te: pd.DataFrame, cols: dict) -> None:
    valores = np.r_[te["y"], *[te[c] for c in cols.values()]]
    folga = 0.03 * (valores.max() - valores.min())
    lim = (valores.min() - folga, valores.max() + folga)

    aplicar_estilo()
    fig, eixos = plt.subplots(1, 2, figsize=(6.3, 3.5), sharex=True, sharey=True)
    for ax, (form, titulo) in zip(eixos, FORMULACOES.items()):
        ax.plot(lim, lim, color=CINZA, linestyle=":", linewidth=1.2, zorder=1)
        # Random Forest vazado e maior, para continuar visível onde as previsões dos
        # dois modelos quase coincidem (formulação em diferença)
        ax.scatter(te["y"], te[cols[("random_forest", form)]], s=48, facecolors="none",
                   edgecolors=AZUL, linewidths=1.4, marker="o", label="Random Forest", zorder=3)
        ax.scatter(te["y"], te[cols[("mlp", form)]], s=16, color=LARANJA, marker="^",
                   linewidths=0, label="MLP", zorder=4)
        ax.set_title(titulo.replace("Formulação em ", "Em "), fontsize=11, loc="left")
        ax.set_xlabel("Observado (pacientes-dia)")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(lambda v, _: milhar(v))
        ax.yaxis.set_major_formatter(lambda v, _: milhar(v))
        ax.tick_params(labelsize=8)
        eixo_limpo(ax, grade="both")
    eixos[0].set_ylabel("Previsto (pacientes-dia)")
    eixos[0].legend(loc="upper left", fontsize=9, frameon=False)
    salvar(fig, "dispersao_teste.png")


def resumo_erros(te: pd.DataFrame, cols: dict, teto: float) -> pd.DataFrame:
    """Erros da previsão média no teste, por configuração e baseline."""
    ingenua = (te["y"] - te["ingenua_mes_anterior"]).abs()
    dias_medio = te["mes"].dt.days_in_month.mean()
    previsoes = {c: c for c in ["ingenua_mes_anterior", "ingenua_mesmo_mes_ano_anterior"]}
    previsoes |= {f"{alg}_{form}": c for (alg, form), c in cols.items()}
    linhas = []
    for nome, c in previsoes.items():
        erro = te[c] - te["y"]
        absoluto = erro.abs()
        linhas.append({
            "configuracao": nome,
            "mae_previsao_media": absoluto.mean(),
            "mae_pct_media_observada": 100 * absoluto.mean() / te["y"].mean(),
            # pacientes-dia por mês / dias do mês = leitos ocupados por dia, em média
            "mae_leitos_por_dia": absoluto.mean() / dias_medio,
            "maior_erro": absoluto.max(),
            "mes_maior_erro": te.loc[absoluto.idxmax(), "mes"].strftime("%Y-%m"),
            "maior_previsao": te[c].max(),
            "meses_previstos_acima_teto_treino": int((te[c] > teto).sum()),
            "meses_melhor_que_ingenua_mes_anterior": int((absoluto < ingenua).sum()),
            "meses_subestimados": int((erro < 0).sum()),
        })
    return pd.DataFrame(linhas)


def main() -> None:
    p = pd.read_parquet(PIPELINE / "etapa5_previsoes.parquet")
    cols = colunas_selecionadas()
    te = p[p["conjunto"] == "teste"].reset_index(drop=True)
    teto = p.loc[p["conjunto"] == "treino", "y"].max()

    TABELAS.mkdir(parents=True, exist_ok=True)
    saida = te[["mes", "y", *cols.values()]].copy()
    saida.to_csv(TABELAS / "previsoes_teste.csv", index=False, float_format="%.0f")
    erros = resumo_erros(te, cols, teto)
    erros.to_csv(TABELAS / "erros_teste.csv", index=False, float_format="%.2f")
    print(f"Maior valor do treino: {milhar(teto)}; meses de teste acima dele: "
          f"{int((te['y'] > teto).sum())} de {len(te)}; média observada no teste: "
          f"{milhar(te['y'].mean())}\n")
    print(erros.round(1).to_string(index=False))

    grafico_serie()
    grafico_previsoes(te, cols, teto)
    grafico_dispersao(te, cols)


if __name__ == "__main__":
    main()
