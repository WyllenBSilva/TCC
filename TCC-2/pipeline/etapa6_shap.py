#!/usr/bin/env python3
"""
etapa6_shap.py
==============
Etapa 6 do pipeline: importância das variáveis por valores SHAP, para os quatro
modelos selecionados na Etapa 5 (Random Forest e MLP, em nível e em diferença).

Uso
---
    python pipeline/etapa6_shap.py

Entradas (somente leitura):
    pipeline/saidas/etapa4_atributos.parquet     116 meses (92 treino, 24 teste)
    pipeline/saidas/etapa5_modelos/*.joblib      10 modelos por configuração

Saídas (em pipeline/saidas/):
    etapa6_shap_valores.parquet   um valor SHAP por configuração, mês e atributo
                                  (média entre sementes) e o valor do atributo
    etapa6_shap_importancia.csv   média de |SHAP| por atributo, no treino e no
                                  teste, com o desvio entre sementes e o sentido
    etapa6_relatorio.txt          resumo e conferências

Método
------
- Valores SHAP intervencionais, com os 92 meses de treino como distribuição de
  referência: o valor-base é a previsão média do modelo no treino, e o SHAP de cada
  atributo é quanto ele afasta a previsão daquele mês desse valor-base.
- Random Forest: TreeExplainer (exato para árvores). MLP: explicador Exact, que
  enumera as 2^p coalizões de atributos (p = 8 em nível, 5 em diferença). Com a mesma
  referência, os dois calculam a mesma quantidade; a conferência compara os dois
  métodos num Random Forest.
- Os valores são calculados para cada uma das 10 sementes e depois tirada a média.
  Como o SHAP intervencional é linear no modelo, a média dos SHAP é o SHAP da
  previsão média, que é a previsão avaliada na Etapa 5.
- Em diferença, o modelo prevê y - lag_1; os valores SHAP explicam essa variação
  mensal, em pacientes-dia. lag_1 não é atributo nessa formulação, só a base a que a
  variação prevista é somada.
- O SHAP descreve o modelo, não a causalidade, e não foi usado para escolher
  atributos (a escolha foi feita antes, pela validação da Etapa 5).
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ    = Path(__file__).resolve().parent.parent
SAIDA   = RAIZ / "pipeline" / "saidas"
ENTRADA = SAIDA / "etapa4_atributos.parquet"
DIR_MOD = SAIDA / "etapa5_modelos"
ARQ_VAL = SAIDA / "etapa6_shap_valores.parquet"
ARQ_IMP = SAIDA / "etapa6_shap_importancia.csv"
ARQ_REL = SAIDA / "etapa6_relatorio.txt"

CONFIGURACOES = [("random_forest", "nivel"), ("random_forest", "diferenca"),
                 ("mlp", "nivel"), ("mlp", "diferenca")]

# tolerância das conferências, em pacientes-dia: o TreeExplainer acumula erro de ponto
# flutuante da ordem de 1e-3 ao somar 300 árvores sobre valores de ~200 mil
TOLERANCIA = 1e-2


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


# ============================================================
# SHAP
# ============================================================

def explicar(modelo, algoritmo: str, X: np.ndarray, Xref: np.ndarray) -> tuple[np.ndarray, float]:
    """Valores SHAP [mês, atributo] e valor-base de um modelo."""
    if algoritmo == "random_forest":
        e = shap.TreeExplainer(modelo, data=Xref, feature_perturbation="interventional")
        return e.shap_values(X, check_additivity=False), float(e.expected_value)
    e = shap.explainers.Exact(modelo.predict, shap.maskers.Independent(Xref, max_samples=len(Xref)))
    r = e(X)
    return r.values, float(np.mean(r.base_values))


def rodar_configuracao(d, algoritmo, formulacao) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pacote = joblib.load(DIR_MOD / f"{algoritmo}_{formulacao}.joblib")
    cols, modelos = pacote["colunas"], pacote["modelos"]
    X = d[cols].to_numpy(float)
    Xref = X[(d["conjunto"] == "treino").to_numpy()]

    valores, bases, erro_aditividade, erro_base = [], [], 0.0, 0.0
    for m in modelos:
        sv, base = explicar(m, algoritmo, X, Xref)
        prev = m.predict(X)
        erro_aditividade = max(erro_aditividade, np.abs(base + sv.sum(axis=1) - prev).max())
        erro_base = max(erro_base, abs(base - m.predict(Xref).mean()))
        valores.append(sv)
        bases.append(base)
    valores = np.stack(valores)                      # [semente, mês, atributo]
    media = valores.mean(axis=0)

    nome = f"{algoritmo}_{formulacao}"
    longo = pd.DataFrame({
        "configuracao": nome,
        "mes": np.repeat(d["mes"].to_numpy(), len(cols)),
        "conjunto": np.repeat(d["conjunto"].to_numpy(), len(cols)),
        "atributo": np.tile(cols, len(d)),
        "valor_atributo": X.ravel(),
        "shap": media.ravel(),
    })

    imp = []
    for conj in ["treino", "teste"]:
        sel = (d["conjunto"] == conj).to_numpy()
        por_semente = np.abs(valores[:, sel, :]).mean(axis=1)    # [semente, atributo]
        for j, c in enumerate(cols):
            x, s = X[sel, j], media[sel, j]
            binario = set(np.unique(X[:, j])) <= {0.0, 1.0}
            imp.append({
                "configuracao": nome, "conjunto": conj, "atributo": c,
                "media_abs_shap": np.abs(s).mean(),
                "dp_entre_sementes": por_semente[:, j].std(ddof=1),
                # sentido: efeito médio quando a binária vale 1; correlação nas demais
                "tipo": "binario" if binario else "continuo",
                "shap_medio_quando_1": s[x == 1].mean() if binario and (x == 1).any() else np.nan,
                "corr_valor_shap": np.corrcoef(x, s)[0, 1] if not binario and x.std() > 0 else np.nan,
            })
    imp = pd.DataFrame(imp)
    imp["posicao"] = imp.groupby("conjunto")["media_abs_shap"].rank(ascending=False).astype(int)

    extra = {"cols": cols, "valores": valores, "base": float(np.mean(bases)),
             "erro_aditividade": erro_aditividade, "erro_base": erro_base, "modelos": modelos,
             "X": X, "Xref": Xref}
    return longo, imp, extra


# ============================================================
# CONFERÊNCIAS
# ============================================================

def conferir(d, extras) -> bool:
    titulo("CONFERÊNCIAS")
    checagens = {}   # item -> maior diferença, em pacientes-dia
    for (alg, form), e in extras.items():
        checagens[f"{alg} {form}: base + soma dos SHAP = previsão"] = e["erro_aditividade"]
        checagens[f"{alg} {form}: base = previsão média no treino"] = e["erro_base"]

    # TreeExplainer e Exact calculam a mesma quantidade (semente 0, 8 meses de teste)
    e = extras[("random_forest", "nivel")]
    m, amostra = e["modelos"][0], e["X"][(d["conjunto"] == "teste").to_numpy()][:8]
    arvore, _ = explicar(m, "random_forest", amostra, e["Xref"])
    exato = shap.explainers.Exact(m.predict, shap.maskers.Independent(e["Xref"], max_samples=len(e["Xref"])))(amostra).values
    checagens["Random Forest: TreeExplainer = Exact"] = np.abs(arvore - exato).max()

    # média dos SHAP entre sementes reproduz a previsão média da Etapa 5
    prev = pd.read_parquet(SAIDA / "etapa5_previsoes.parquet")
    res = pd.read_csv(SAIDA / "etapa5_resultados.csv")
    for (alg, form), e in extras.items():
        conj = res[(res["modelo"] == alg) & (res["formulacao"] == form) & res["selecionado"]]["atributos"].iloc[0]
        alvo = prev[f"{alg}_{form}_{conj}"].to_numpy()
        base_y = d["lag_1"].to_numpy(float) if form == "diferenca" else 0.0
        reconstruida = base_y + e["base"] + e["valores"].mean(axis=0).sum(axis=1)
        checagens[f"{alg} {form}: SHAP médio reproduz a previsão da Etapa 5"] = \
            np.abs(reconstruida - alvo).max()

    tudo_ok = True
    log(f"  {'':<66}{'maior dif.':>11}")
    for item, dif in checagens.items():
        ok = dif < TOLERANCIA
        tudo_ok &= bool(ok)
        log(f"  {item:<66}{dif:>11.1e}  {'OK' if ok else 'FALHOU'}")
    log(f"\n  tolerância: {TOLERANCIA} pacientes-dia")
    return tudo_ok


# ============================================================
# RELATÓRIO
# ============================================================

def relatar(imp: pd.DataFrame, extras) -> None:
    for (alg, form), e in extras.items():
        nome = f"{alg}_{form}"
        titulo(f"{alg} {form} — valor-base {e['base']:,.0f} pacientes-dia"
               + (" (variação mensal prevista)" if form == "diferenca" else ""))
        t = imp[imp["configuracao"] == nome].pivot(index="atributo", columns="conjunto")
        t = t.sort_values(("media_abs_shap", "teste"), ascending=False)
        log(f"  {'atributo':<14}{'|SHAP| teste':>13}{'± dp':>8}{'|SHAP| treino':>15}"
            f"{'pos. teste':>11}{'pos. treino':>12}{'SHAP se =1':>12}{'corr':>7}")
        for a, r in t.iterrows():
            quando_1 = r[("shap_medio_quando_1", "teste")]
            corr = r[("corr_valor_shap", "teste")]
            log(f"  {a:<14}{r[('media_abs_shap', 'teste')]:>13,.0f}{r[('dp_entre_sementes', 'teste')]:>8,.0f}"
                f"{r[('media_abs_shap', 'treino')]:>15,.0f}{int(r[('posicao', 'teste')]):>11}"
                f"{int(r[('posicao', 'treino')]):>12}"
                f"{'' if pd.isna(quando_1) else f'{quando_1:,.0f}':>12}"
                f"{'' if pd.isna(corr) else f'{corr:+.2f}':>7}")
    log("\n  |SHAP|: média do valor absoluto nos meses do conjunto, em pacientes-dia")
    log("  SHAP se =1: média do SHAP nos meses de teste em que a binária vale 1")
    log("  corr: correlação, no teste, entre o valor do atributo e o seu SHAP")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    faltando = [c for c in CONFIGURACOES if not (DIR_MOD / f"{c[0]}_{c[1]}.joblib").is_file()]
    if faltando or not ENTRADA.is_file():
        sys.exit("Modelos ou atributos não encontrados. Rodar as Etapas 4 e 5 antes.")

    titulo("ETAPA 6 — Importância das variáveis (SHAP)")
    log(f"shap {shap.__version__}; referência: meses de treino; média de 10 sementes")
    d = pd.read_parquet(ENTRADA)

    longos, imps, extras = [], [], {}
    for alg, form in CONFIGURACOES:
        t0 = time.time()
        longo, imp, extra = rodar_configuracao(d, alg, form)
        longos.append(longo)
        imps.append(imp)
        extras[(alg, form)] = extra
        log(f"  {alg:<14} {form:<10} {len(extra['cols'])} atributos  ({time.time() - t0:.0f} s)")

    imp = pd.concat(imps, ignore_index=True)
    relatar(imp, extras)
    tudo_ok = conferir(d, extras)

    pd.concat(longos, ignore_index=True).to_parquet(ARQ_VAL, index=False)
    imp.to_csv(ARQ_IMP, index=False)
    log(f"\nGravados: {ARQ_VAL.name} e {ARQ_IMP.name}")
    ARQ_REL.write_text("\n".join(_linhas), encoding="utf-8")
    print(f"Relatório: {ARQ_REL.relative_to(RAIZ)}")
    sys.exit(0 if tudo_ok else 1)


if __name__ == "__main__":
    main()
