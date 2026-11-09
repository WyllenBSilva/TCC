#!/usr/bin/env python3
"""
etapa5_modelagem.py
===================
Etapa 5 do pipeline: baselines ingênuas, Random Forest e MLPRegressor nas
formulações em nível e em diferença, com hiperparâmetros escolhidos por validação
cronológica dentro do treino e avaliação final no teste.

Uso
---
    python pipeline/etapa5_modelagem.py

Entrada (somente leitura):
    pipeline/saidas/etapa4_atributos.parquet      116 meses (92 treino, 24 teste)

Saídas (em pipeline/saidas/):
    etapa5_resultados.csv        uma linha por configuração e baseline: validação,
                                 treino e teste (média e desvio entre sementes)
    etapa5_validacao.csv         erro de validação de cada combinação de
                                 hiperparâmetros, por dobra
    etapa5_previsoes.parquet     previsão média entre sementes, mês a mês
    etapa5_modelos/              modelos finais das configurações selecionadas,
                                 um por semente (para o SHAP)
    etapa5_relatorio.txt         resumo, escolhas e conferências

Desenho experimental
--------------------
- Previsão de um mês à frente. Formulação em nível: alvo y, atributos de calendário
  e defasagens. Formulação em diferença: alvo y - lag_1, só calendário; a previsão
  de y é lag_1 + previsão da diferença.
- Para cada algoritmo e formulação, dois conjuntos de atributos: o base e o base com
  as proporções defasadas. O conjunto mantido é o de menor erro na validação.
- Validação: TimeSeriesSplit com 4 dobras de 12 meses, só dentro do treino. Cada
  combinação de hiperparâmetros é avaliada com SEMENTES_VALIDACAO e escolhida pelo
  menor MAE médio. O teste não participa de nenhuma escolha.
- Modelo final: refeito no treino inteiro com cada uma das SEMENTES_FINAIS; as
  métricas são a média e o desvio-padrão entre sementes.
- MLP: atributos e alvo padronizados (TransformedTargetRegressor), nas duas
  formulações.
"""

import sys
import time
import warnings
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from etapa4_atributos import CALENDARIO, LAGS, PROPORCOES


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ     = Path(__file__).resolve().parent.parent
SAIDA    = RAIZ / "pipeline" / "saidas"
ENTRADA  = SAIDA / "etapa4_atributos.parquet"
ARQ_RES  = SAIDA / "etapa5_resultados.csv"
ARQ_VAL  = SAIDA / "etapa5_validacao.csv"
ARQ_PREV = SAIDA / "etapa5_previsoes.parquet"
DIR_MOD  = SAIDA / "etapa5_modelos"
ARQ_REL  = SAIDA / "etapa5_relatorio.txt"

SEMENTES_VALIDACAO = [0, 1, 2]
SEMENTES_FINAIS    = list(range(10))
N_DOBRAS, TAM_DOBRA = 4, 12
N_PROCESSOS = 3

FORMULACOES = {
    "nivel":     {"alvo": "y",      "atributos": CALENDARIO + LAGS},
    "diferenca": {"alvo": "y_diff", "atributos": CALENDARIO},
}
CONJUNTOS = {"base": [], "com_proporcoes": PROPORCOES}

GRADES = {
    "random_forest": {
        "n_estimators": [100, 300],
        "max_depth": [None, 4, 8],
        "min_samples_split": [2, 5, 10],
    },
    "mlp": {
        "hidden_layer_sizes": [(8,), (16,), (16, 8)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-3, 1e-1],
        "learning_rate_init": [1e-3, 1e-2],
    },
}


# ============================================================
# UTILITÁRIOS
# ============================================================

_linhas: list[str] = []


def log(texto="") -> None:
    """Imprime e acumula o texto para gravar no relatório."""
    print(texto, flush=True)
    _linhas.append(str(texto))


def titulo(texto: str) -> None:
    log("\n" + "=" * 74)
    log(texto)
    log("=" * 74)


def combinacoes(grade: dict) -> list[dict]:
    return [dict(zip(grade, v)) for v in product(*grade.values())]


def metricas(y, p) -> dict:
    return {
        "mae": mean_absolute_error(y, p),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": r2_score(y, p),
        "vies": float(np.mean(p - y)),
    }


# ============================================================
# MODELOS
# ============================================================

def criar(algoritmo: str, params: dict, semente: int):
    if algoritmo == "random_forest":
        return RandomForestRegressor(random_state=semente, n_jobs=1, **params)
    mlp = MLPRegressor(random_state=semente, solver="adam", max_iter=5000, **params)
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), mlp),
                                      transformer=StandardScaler())


def ajustar(algoritmo, params, semente, X, y):
    """Ajusta e devolve (modelo, avisou_convergencia)."""
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        m = criar(algoritmo, params, semente).fit(X, y)
    return m, any(issubclass(a.category, ConvergenceWarning) for a in avisos)


def validar_combinacao(algoritmo, params, X, y) -> np.ndarray:
    """MAE de validação [semente, dobra]. Em diferença, o erro em y_diff é igual ao
    erro em y, pois y - (lag_1 + p) = y_diff - p."""
    dobras = TimeSeriesSplit(n_splits=N_DOBRAS, test_size=TAM_DOBRA).split(X)
    dobras = list(dobras)
    erros = np.zeros((len(SEMENTES_VALIDACAO), N_DOBRAS))
    for i, s in enumerate(SEMENTES_VALIDACAO):
        for j, (tr, va) in enumerate(dobras):
            m, _ = ajustar(algoritmo, params, s, X[tr], y[tr])
            erros[i, j] = mean_absolute_error(y[va], m.predict(X[va]))
    return erros


# ============================================================
# EXPERIMENTO
# ============================================================

def rodar_configuracao(d, algoritmo, formulacao, conjunto) -> tuple[dict, pd.DataFrame, dict]:
    f = FORMULACOES[formulacao]
    cols = f["atributos"] + CONJUNTOS[conjunto]
    tr, te = d[d["conjunto"] == "treino"], d[d["conjunto"] == "teste"]
    Xtr, ytr = tr[cols].to_numpy(float), tr[f["alvo"]].to_numpy(float)
    Xte = te[cols].to_numpy(float)

    # 1. escolha dos hiperparâmetros, só no treino
    combos = combinacoes(GRADES[algoritmo])
    erros = joblib.Parallel(n_jobs=N_PROCESSOS)(
        joblib.delayed(validar_combinacao)(algoritmo, p, Xtr, ytr) for p in combos)
    val = pd.DataFrame([{**{"params": str(p)}, "mae_validacao": e.mean(),
                         **{f"mae_dobra_{j + 1}": e[:, j].mean() for j in range(N_DOBRAS)}}
                        for p, e in zip(combos, erros)])
    melhor = int(val["mae_validacao"].idxmin())
    params = combos[melhor]

    # 2. modelo final no treino inteiro, uma vez por semente
    base_tr = tr["lag_1"].to_numpy(float) if formulacao == "diferenca" else 0.0
    base_te = te["lag_1"].to_numpy(float) if formulacao == "diferenca" else 0.0
    prev_tr, prev_te, modelos, nao_convergiu = [], [], [], 0
    for s in SEMENTES_FINAIS:
        m, aviso = ajustar(algoritmo, params, s, Xtr, ytr)
        nao_convergiu += aviso
        prev_tr.append(base_tr + m.predict(Xtr))
        prev_te.append(base_te + m.predict(Xte))
        modelos.append(m)

    linha = {
        "modelo": algoritmo, "formulacao": formulacao, "atributos": conjunto,
        "n_atributos": len(cols), "hiperparametros": str(params),
        "mae_validacao": val.at[melhor, "mae_validacao"],
        **{f"mae_dobra_{j + 1}": val.at[melhor, f"mae_dobra_{j + 1}"] for j in range(N_DOBRAS)},
        "sementes_sem_convergencia": nao_convergiu,
    }
    for nome, y, prev in [("treino", tr["y"].to_numpy(float), prev_tr),
                          ("teste", te["y"].to_numpy(float), prev_te)]:
        por_semente = pd.DataFrame([metricas(y, p) for p in prev])
        for k in por_semente:
            linha[f"{k}_{nome}"] = por_semente[k].mean()
            linha[f"{k}_{nome}_dp"] = por_semente[k].std(ddof=1)

    val.insert(0, "modelo", algoritmo)
    val.insert(1, "formulacao", formulacao)
    val.insert(2, "atributos", conjunto)
    extra = {"colunas": cols, "modelos": modelos,
             "prev_tr": np.mean(prev_tr, axis=0), "prev_te": np.mean(prev_te, axis=0)}
    return linha, val, extra


def baselines(d) -> list[dict]:
    tr, te = d[d["conjunto"] == "treino"], d[d["conjunto"] == "teste"]
    linhas = []
    for nome, col in [("ingenua_mes_anterior", "lag_1"), ("ingenua_mesmo_mes_ano_anterior", "lag_12")]:
        linha = {"modelo": nome, "formulacao": "-", "atributos": "-", "selecionado": True}
        for conj, x in [("treino", tr), ("teste", te)]:
            for k, v in metricas(x["y"].to_numpy(float), x[col].to_numpy(float)).items():
                linha[f"{k}_{conj}"] = v
        linhas.append(linha)
    return linhas


# ============================================================
# CONFERÊNCIAS
# ============================================================

def conferir(d, resultados, extras) -> bool:
    titulo("CONFERÊNCIAS")
    tr, te = d[d["conjunto"] == "treino"], d[d["conjunto"] == "teste"]
    ult_validacao = max(va.max() for _, va in
                        TimeSeriesSplit(n_splits=N_DOBRAS, test_size=TAM_DOBRA).split(tr))
    checagens = {
        "meses de treino = 92": len(tr) == 92,
        "meses de teste = 24": len(te) == 24,
        "treino termina antes do teste": tr["mes"].max() < te["mes"].min(),
        "validação usa só meses de treino": ult_validacao == len(tr) - 1,
        "sem valores nulos nos atributos": not d.isna().any().any(),
        "baseline mês anterior = média |y - lag_1| no teste": np.isclose(
            resultados.loc[resultados["modelo"] == "ingenua_mes_anterior", "mae_teste"].iloc[0],
            (te["y"] - te["lag_1"]).abs().mean()),
    }
    # mesma semente, mesmo resultado
    chave = next(iter(extras))
    alg, form, conj = chave
    linha = resultados[(resultados["modelo"] == alg) & (resultados["formulacao"] == form)
                       & (resultados["atributos"] == conj)].iloc[0]
    params = eval(linha["hiperparametros"])
    cols = extras[chave]["colunas"]
    alvo = FORMULACOES[form]["alvo"]
    m1, _ = ajustar(alg, params, 0, tr[cols].to_numpy(float), tr[alvo].to_numpy(float))
    m2 = extras[chave]["modelos"][0]
    checagens["reajuste com a mesma semente reproduz a previsão"] = np.allclose(
        m1.predict(te[cols].to_numpy(float)), m2.predict(te[cols].to_numpy(float)))

    tudo_ok = True
    for item, ok in checagens.items():
        tudo_ok &= bool(ok)
        log(f"  {item:<55} {'OK' if ok else 'FALHOU'}")
    return tudo_ok


# ============================================================
# RELATÓRIO
# ============================================================

def relatar(res: pd.DataFrame) -> None:
    titulo("Escolha do conjunto de atributos (menor MAE de validação)")
    modelos = res[~res["modelo"].str.startswith("ingenua")]
    for (alg, form), g in modelos.groupby(["modelo", "formulacao"], sort=False):
        partes = "  |  ".join(f"{r.atributos}: {r.mae_validacao:,.0f}" for r in g.itertuples())
        escolhido = g.loc[g["selecionado"], "atributos"].iloc[0]
        log(f"  {alg:<14} {form:<10} {partes}  ->  {escolhido}")

    titulo("Hiperparâmetros escolhidos")
    for r in modelos[modelos["selecionado"]].itertuples():
        log(f"  {r.modelo:<14} {r.formulacao:<10} {r.hiperparametros}")
        if r.sementes_sem_convergencia:
            log(f"  {'':<25} sem convergência em {int(r.sementes_sem_convergencia)} de "
                f"{len(SEMENTES_FINAIS)} sementes")

    titulo("MAE de validação por dobra (configurações selecionadas)")
    log(f"  {'':<26}{'dobra 1':>10}{'dobra 2':>10}{'dobra 3':>10}{'dobra 4':>10}")
    log(f"  {'(meses de validação)':<26}{'19/09-20/08':>12}{'20/09-21/08':>12}"
        f"{'21/09-22/08':>12}{'22/09-23/08':>12}")
    for r in modelos[modelos["selecionado"]].itertuples():
        log(f"  {r.modelo + ' ' + r.formulacao:<26}" + "".join(
            f"{getattr(r, f'mae_dobra_{j}'):>10,.0f}" for j in range(1, N_DOBRAS + 1)))

    titulo("Resultados (média entre sementes; baselines são determinísticas)")
    sel = res[res["selecionado"]]
    log(f"  {'':<34}{'MAE trein.':>11}{'MAE teste':>11}{'± dp':>8}{'RMSE teste':>12}"
        f"{'R² teste':>10}{'viés teste':>12}")
    for r in sel.itertuples():
        nome = r.modelo if r.formulacao == "-" else f"{r.modelo} {r.formulacao}"
        dp = "" if pd.isna(getattr(r, "mae_teste_dp", np.nan)) else f"{r.mae_teste_dp:,.0f}"
        log(f"  {nome:<34}{r.mae_treino:>11,.0f}{r.mae_teste:>11,.0f}{dp:>8}"
            f"{r.rmse_teste:>12,.0f}{r.r2_teste:>10.3f}{r.vies_teste:>12,.0f}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not ENTRADA.is_file():
        sys.exit(f"Arquivo de entrada não encontrado: {ENTRADA}. Rodar a Etapa 4 antes.")

    titulo("ETAPA 5 — Modelagem e avaliação")
    log(f"Entrada: {ENTRADA.relative_to(RAIZ)}")
    log(f"Validação: TimeSeriesSplit, {N_DOBRAS} dobras de {TAM_DOBRA} meses, sementes "
        f"{SEMENTES_VALIDACAO}; modelo final com {len(SEMENTES_FINAIS)} sementes")
    d = pd.read_parquet(ENTRADA)

    linhas, validacoes, extras = baselines(d), [], {}
    for alg, form, conj in product(GRADES, FORMULACOES, CONJUNTOS):
        t0 = time.time()
        linha, val, extra = rodar_configuracao(d, alg, form, conj)
        linhas.append(linha)
        validacoes.append(val)
        extras[(alg, form, conj)] = extra
        log(f"  {alg:<14} {form:<10} {conj:<15} validação {linha['mae_validacao']:>9,.0f}  "
            f"teste {linha['mae_teste']:>9,.0f}  ({time.time() - t0:.0f} s)")

    res = pd.DataFrame(linhas)
    modelos = ~res["modelo"].str.startswith("ingenua")
    menor = res[modelos].groupby(["modelo", "formulacao"])["mae_validacao"].transform("min")
    res.loc[modelos, "selecionado"] = res.loc[modelos, "mae_validacao"] == menor
    res["selecionado"] = res["selecionado"].astype(bool)

    relatar(res)
    tudo_ok = conferir(d, res, extras)

    res.to_csv(ARQ_RES, index=False)
    pd.concat(validacoes, ignore_index=True).to_csv(ARQ_VAL, index=False)

    prev = d[["mes", "conjunto", "y", "lag_1", "lag_12"]].rename(
        columns={"lag_1": "ingenua_mes_anterior", "lag_12": "ingenua_mesmo_mes_ano_anterior"})
    DIR_MOD.mkdir(exist_ok=True)
    for (alg, form, conj), e in extras.items():
        nome = f"{alg}_{form}_{conj}"
        prev[nome] = np.r_[e["prev_tr"], e["prev_te"]]
        sel = res[(res["modelo"] == alg) & (res["formulacao"] == form)
                  & (res["atributos"] == conj)]["selecionado"].iloc[0]
        if sel:
            joblib.dump({"colunas": e["colunas"], "formulacao": form, "modelos": e["modelos"]},
                        DIR_MOD / f"{alg}_{form}.joblib")
    prev.to_parquet(ARQ_PREV, index=False)

    log(f"\nGravados: {ARQ_RES.name}, {ARQ_VAL.name}, {ARQ_PREV.name} e {DIR_MOD.name}/")
    ARQ_REL.write_text("\n".join(_linhas), encoding="utf-8")
    print(f"Relatório: {ARQ_REL.relative_to(RAIZ)}")
    sys.exit(0 if tudo_ok else 1)


if __name__ == "__main__":
    main()
