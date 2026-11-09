#!/usr/bin/env python3
"""
verifica_resultados.py
======================
Verificações dos resultados das Etapas 5 e 6, a partir das saídas gravadas e dos
modelos salvos, mais duas análises complementares.

Uso
---
    python tests/verifica_resultados.py

Entradas (somente leitura):
    pipeline/saidas/etapa3_serie_mensal.parquet
    pipeline/saidas/etapa4_atributos.parquet
    pipeline/saidas/etapa5_resultados.csv, etapa5_validacao.csv, etapa5_previsoes.parquet
    pipeline/saidas/etapa5_modelos/*.joblib
    pipeline/saidas/etapa6_shap_valores.parquet

Saídas (em tests/saidas/):
    relatorio_verifica_resultados.txt
    validacao_media_sementes.csv     MAE de validação por dobra: média dos modelos
                                     individuais x previsão média de 10 sementes
    sensibilidade_sem_reconstrucao.csv  teste com o alvo sem reconstrução das lacunas

Verificações
------------
1. O teste não participa das escolhas: as dobras usam só meses de treino, e os
   hiperparâmetros e conjuntos escolhidos são os de menor erro em etapa5_validacao.csv.
2. lag_1 de cada mês (inclusive no teste) é o valor observado do mês anterior, e não
   uma previsão: o teste são 24 previsões de um mês à frente, não uma previsão
   encadeada de 24 meses.
3. As padronizações do MLP (X e y) são ajustadas só com os meses de ajuste da dobra.
4. As métricas da formulação em diferença são calculadas sobre y = lag_1 + previsão.
5. Os valores SHAP do MLP estão em pacientes-dia (nível) ou em variação mensal de
   pacientes-dia (diferença): base + soma reproduz a previsão nessa escala.

Análises complementares
-----------------------
A. Validação da previsão média de 10 sementes (o que as figuras mostram), nas mesmas
   dobras, com os hiperparâmetros escolhidos.
B. Alvo sem a reconstrução das lacunas: os atributos são refeitos com
   pacientes_dia_sem_reconstrucao e os modelos escolhidos são reajustados com os
   MESMOS hiperparâmetros (a busca em grade não é refeita).
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "pipeline"))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from etapa4_atributos import montar
from etapa5_modelagem import (FORMULACOES, N_DOBRAS, SEMENTES_FINAIS, TAM_DOBRA, ajustar,
                              criar, metricas)


SAIDA_PIPE = RAIZ / "pipeline" / "saidas"
SAIDA      = RAIZ / "tests" / "saidas"
N_PROCESSOS = 3

_linhas: list[str] = []


def log(texto="") -> None:
    print(texto, flush=True)
    _linhas.append(str(texto))


def titulo(texto: str) -> None:
    log("\n" + "=" * 74)
    log(texto)
    log("=" * 74)


def selecionadas() -> pd.DataFrame:
    r = pd.read_csv(SAIDA_PIPE / "etapa5_resultados.csv")
    return r[r["selecionado"] & ~r["modelo"].str.startswith("ingenua")].reset_index(drop=True)


def dobras(n: int):
    return list(TimeSeriesSplit(n_splits=N_DOBRAS, test_size=TAM_DOBRA).split(np.arange(n)))


def separar(d, form):
    cols = FORMULACOES[form]["atributos"]
    alvo = FORMULACOES[form]["alvo"]
    tr, te = d[d["conjunto"] == "treino"], d[d["conjunto"] == "teste"]
    return tr, te, cols, alvo


# ============================================================
# VERIFICAÇÕES
# ============================================================

def v1_teste_fora_das_escolhas(d, sel) -> dict:
    titulo("1. O teste não participa das escolhas")
    tr = d[d["conjunto"] == "treino"]
    ultimo = max(va.max() for _, va in dobras(len(tr)))
    val = pd.read_csv(SAIDA_PIPE / "etapa5_validacao.csv")
    res = pd.read_csv(SAIDA_PIPE / "etapa5_resultados.csv")
    res = res[~res["modelo"].str.startswith("ingenua")]

    hiper_ok = True
    for r in res.itertuples():
        g = val[(val["modelo"] == r.modelo) & (val["formulacao"] == r.formulacao)
                & (val["atributos"] == r.atributos)]
        hiper_ok &= g.loc[g["mae_validacao"].idxmin(), "params"] == r.hiperparametros
    conj_ok = True
    for (alg, form), g in res.groupby(["modelo", "formulacao"]):
        conj_ok &= g.loc[g["mae_validacao"].idxmin(), "atributos"] == g.loc[g["selecionado"], "atributos"].iloc[0]

    log(f"  último índice de validação: {ultimo} (treino tem {len(tr)} meses, índices 0 a {len(tr) - 1})")
    log(f"  último mês de validação: {tr['mes'].iloc[ultimo]:%Y-%m}; primeiro mês de teste: "
        f"{d.loc[d['conjunto'] == 'teste', 'mes'].min():%Y-%m}")
    return {
        "dobras de validação dentro do treino": ultimo == len(tr) - 1,
        "hiperparâmetros = menor MAE de validação (8 configurações)": hiper_ok,
        "conjunto de atributos = menor MAE de validação (4 pares)": conj_ok,
    }


def v2_lag1_observado(d) -> dict:
    titulo("2. lag_1 é o valor observado do mês anterior")
    s = pd.read_parquet(SAIDA_PIPE / "etapa3_serie_mensal.parquet").set_index("mes")["pacientes_dia"]
    anterior = s.reindex(d["mes"] - pd.DateOffset(months=1)).to_numpy()
    iguais = d["lag_1"].to_numpy() == anterior
    te = d["conjunto"] == "teste"
    p = pd.read_parquet(SAIDA_PIPE / "etapa5_previsoes.parquet")
    log(f"  meses com lag_1 = y observado do mês anterior: {iguais.sum()} de {len(d)}")
    log(f"  primeiro mês de teste {d.loc[te, 'mes'].iloc[0]:%Y-%m}: lag_1 = "
        f"{d.loc[te, 'lag_1'].iloc[0]:,} = y de 2023-08 ({s['2023-08-01']:,})")
    log("  o lag_1 de cada mês de teste é o valor observado, nunca a previsão do mês")
    log("  anterior: o teste são 24 previsões independentes de um mês à frente")
    return {
        "lag_1 = y observado do mês anterior (116 meses)": bool(iguais.all()),
        "baseline mês anterior usa o mesmo lag_1": bool(
            (p["ingenua_mes_anterior"].to_numpy() == d["lag_1"].to_numpy()).all()),
    }


def v3_padronizacao_por_dobra(d, sel) -> dict:
    titulo("3. Padronização do MLP ajustada só nos meses de ajuste")
    checagens = {}
    for r in sel[sel["modelo"] == "mlp"].itertuples():
        tr, _, cols, alvo = separar(d, r.formulacao)
        X, y = tr[cols].to_numpy(float), tr[alvo].to_numpy(float)
        ok = True
        for j, (ajuste, _) in enumerate(dobras(len(tr))):
            m = criar("mlp", eval(r.hiperparametros), 0).fit(X[ajuste], y[ajuste])
            sx = m.regressor_.named_steps["standardscaler"]
            sy = m.transformer_
            ok &= np.allclose(sx.mean_, X[ajuste].mean(axis=0)) and np.allclose(sx.scale_, X[ajuste].std(axis=0))
            ok &= np.isclose(sy.mean_[0], y[ajuste].mean()) and np.isclose(sy.scale_[0], y[ajuste].std())
            if j == 0:
                log(f"  mlp {r.formulacao}, dobra 1: média de y usada na padronização "
                    f"{sy.mean_[0]:,.1f} = média dos {len(ajuste)} meses de ajuste "
                    f"({y[ajuste].mean():,.1f}); média dos 92 meses: {y.mean():,.1f}")
        checagens[f"mlp {r.formulacao}: média e escala de X e y = meses de ajuste (4 dobras)"] = ok
    return checagens


def v4_metricas_diferenca(d, sel) -> dict:
    titulo("4. Métricas em diferença calculadas sobre y = lag_1 + previsão")
    checagens = {}
    for r in sel[sel["formulacao"] == "diferenca"].itertuples():
        pacote = joblib.load(SAIDA_PIPE / "etapa5_modelos" / f"{r.modelo}_diferenca.joblib")
        te = d[d["conjunto"] == "teste"]
        X = te[pacote["colunas"]].to_numpy(float)
        mae_y = [mean_absolute_error(te["y"], te["lag_1"] + m.predict(X)) for m in pacote["modelos"]]
        mae_d = [mean_absolute_error(te["y_diff"], m.predict(X)) for m in pacote["modelos"]]
        log(f"  {r.modelo} diferença: MAE recalculado em y {np.mean(mae_y):,.1f}; "
            f"relatório {r.mae_teste:,.1f}; MAE em y_diff {np.mean(mae_d):,.1f}")
        checagens[f"{r.modelo} diferença: MAE do relatório = MAE sobre y reconstruído"] = \
            np.isclose(np.mean(mae_y), r.mae_teste)
    return checagens


def v5_escala_shap(d, sel) -> dict:
    titulo("5. Escala dos valores SHAP")
    v = pd.read_parquet(SAIDA_PIPE / "etapa6_shap_valores.parquet")
    p = pd.read_parquet(SAIDA_PIPE / "etapa5_previsoes.parquet").set_index("mes")
    checagens = {}
    for r in sel.itertuples():
        nome = f"{r.modelo}_{r.formulacao}"
        soma = v[v["configuracao"] == nome].groupby("mes")["shap"].sum()
        prev = p[f"{nome}_{r.atributos}"].reindex(soma.index)
        lag1 = d.set_index("mes")["lag_1"].reindex(soma.index)
        treino = (p["conjunto"] == "treino").reindex(soma.index).to_numpy()
        # valor-base = previsão média do modelo no treino, na escala que ele prevê
        saida = prev - lag1 if r.formulacao == "diferenca" else prev
        base = saida[treino].mean()
        erro = np.abs(base + soma - saida).max()
        escala = "variação mensal (pacientes-dia)" if r.formulacao == "diferenca" else "pacientes-dia"
        log(f"  {nome:<24} valor-base {base:>10,.0f}  maior |base + soma - previsão| "
            f"{erro:.1e}  escala: {escala}")
        checagens[f"{nome}: base + soma dos SHAP = previsão em {escala}"] = erro < 0.01
    return checagens


# ============================================================
# ANÁLISES COMPLEMENTARES
# ============================================================

def _validar_sementes(alg, params, X, y):
    """Por dobra: MAE médio dos modelos individuais e MAE da previsão média."""
    linhas = []
    for j, (ajuste, va) in enumerate(dobras(len(y))):
        prev = np.array([ajustar(alg, params, s, X[ajuste], y[ajuste])[0].predict(X[va])
                         for s in SEMENTES_FINAIS])
        linhas.append({"dobra": j + 1,
                       "mae_individual_medio": np.mean([mean_absolute_error(y[va], p) for p in prev]),
                       "mae_previsao_media": mean_absolute_error(y[va], prev.mean(axis=0))})
    return linhas


def a_validacao_media(d, sel) -> pd.DataFrame:
    titulo("A. Validação da previsão média de 10 sementes (mesmas dobras e hiperparâmetros)")
    tarefas = []
    for r in sel.itertuples():
        tr, _, cols, alvo = separar(d, r.formulacao)
        tarefas.append((r, tr[cols].to_numpy(float), tr[alvo].to_numpy(float)))
    saidas = joblib.Parallel(n_jobs=N_PROCESSOS)(
        joblib.delayed(_validar_sementes)(r.modelo, eval(r.hiperparametros), X, y)
        for r, X, y in tarefas)
    t = pd.concat([pd.DataFrame(s).assign(configuracao=f"{r.modelo}_{r.formulacao}")
                   for (r, _, _), s in zip(tarefas, saidas)], ignore_index=True)
    resumo = t.groupby("configuracao", sort=False)[["mae_individual_medio", "mae_previsao_media"]].mean()
    log(f"  {'configuração':<26}{'individual':>12}{'média de 10':>13}   por dobra (média de 10)")
    for nome, g in t.groupby("configuracao", sort=False):
        log(f"  {nome:<26}{resumo.at[nome, 'mae_individual_medio']:>12,.0f}"
            f"{resumo.at[nome, 'mae_previsao_media']:>13,.0f}   "
            + " / ".join(f"{x:,.0f}" for x in g["mae_previsao_media"]))
    log("  individual: média, entre 10 sementes e 4 dobras, do MAE de cada modelo")
    log("  (a Etapa 5 usou 3 sementes, por isso o valor difere um pouco do relatório)")
    return t


def b_sem_reconstrucao(sel) -> pd.DataFrame:
    titulo("B. Alvo sem a reconstrução das lacunas (mesmos hiperparâmetros)")
    s = pd.read_parquet(SAIDA_PIPE / "etapa3_serie_mensal.parquet")
    alt = s.assign(pacientes_dia=s["pacientes_dia_sem_reconstrucao"])
    variantes = {"com reconstrução": montar(s), "sem reconstrução": montar(alt)}
    linhas = []
    for variante, d in variantes.items():
        tr, te = d[d["conjunto"] == "treino"], d[d["conjunto"] == "teste"]
        for nome, col in [("ingenua_mes_anterior", "lag_1"), ("ingenua_mesmo_mes_ano_anterior", "lag_12")]:
            linhas.append({"variante": variante, "configuracao": nome,
                           **{f"{k}_teste": v for k, v in metricas(te["y"], te[col]).items()}})
        for r in sel.itertuples():
            _, _, cols, alvo = separar(d, r.formulacao)
            base = te["lag_1"].to_numpy(float) if r.formulacao == "diferenca" else 0.0
            por_semente = []
            for sem in SEMENTES_FINAIS:
                m, _ = ajustar(r.modelo, eval(r.hiperparametros), sem,
                               tr[cols].to_numpy(float), tr[alvo].to_numpy(float))
                por_semente.append(metricas(te["y"].to_numpy(float), base + m.predict(te[cols].to_numpy(float))))
            ps = pd.DataFrame(por_semente)
            linhas.append({"variante": variante, "configuracao": f"{r.modelo}_{r.formulacao}",
                           **{f"{k}_teste": ps[k].mean() for k in ps}, "mae_teste_dp": ps["mae"].std(ddof=1)})
    t = pd.DataFrame(linhas)
    piv = t.pivot(index="configuracao", columns="variante", values="mae_teste").reindex(t["configuracao"].unique())
    d0, d1 = variantes["com reconstrução"], variantes["sem reconstrução"]
    dif = (d0["y"] - d1["y"])
    log(f"  diferença no alvo (com - sem): {int(dif.sum()):,} pacientes-dia em {int((dif != 0).sum())} "
        f"dos 116 meses; no teste: {int(dif[d0['conjunto'] == 'teste'].sum()):,}")
    log(f"  {'configuração':<34}{'MAE com':>10}{'MAE sem':>10}{'diferença':>11}")
    for nome, x in piv.iterrows():
        log(f"  {nome:<34}{x['com reconstrução']:>10,.0f}{x['sem reconstrução']:>10,.0f}"
            f"{x['sem reconstrução'] - x['com reconstrução']:>+11,.0f}")
    return t


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    d = pd.read_parquet(SAIDA_PIPE / "etapa4_atributos.parquet")
    sel = selecionadas()

    checagens = {}
    for v in (v1_teste_fora_das_escolhas, v2_lag1_observado):
        checagens |= v(d, sel) if v is v1_teste_fora_das_escolhas else v(d)
    for v in (v3_padronizacao_por_dobra, v4_metricas_diferenca, v5_escala_shap):
        checagens |= v(d, sel)

    validacao = a_validacao_media(d, sel)
    sensibilidade = b_sem_reconstrucao(sel)

    titulo("CONFERÊNCIAS")
    tudo_ok = True
    for item, ok in checagens.items():
        tudo_ok &= bool(ok)
        log(f"  {item:<68} {'OK' if ok else 'FALHOU'}")

    validacao.to_csv(SAIDA / "validacao_media_sementes.csv", index=False, float_format="%.2f")
    sensibilidade.to_csv(SAIDA / "sensibilidade_sem_reconstrucao.csv", index=False, float_format="%.4f")
    (SAIDA / "relatorio_verifica_resultados.txt").write_text("\n".join(_linhas), encoding="utf-8")
    sys.exit(0 if tudo_ok else 1)


if __name__ == "__main__":
    main()
