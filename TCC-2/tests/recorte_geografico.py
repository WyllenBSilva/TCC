#!/usr/bin/env python3
"""
recorte_geografico.py
=====================
Verifica o recorte geográfico da base: os arquivos RD de SC são organizados pelo
local de internação (MUNIC_MOV), e não pela residência do paciente (MUNIC_RES).
Mede quanto da base corresponde a pacientes residentes em outras UFs.

Uso
---
    python tests/recorte_geografico.py

Entrada (somente leitura):
    pipeline/saidas/etapa1_base_limpa.parquet

Saída (em tests/saidas/):
    relatorio_recorte_geografico.txt   registros e dias de permanência por UF de
                                       internação e de residência, e participação
                                       anual dos não residentes

Os dois primeiros dígitos do código IBGE do município identificam a UF (42 = SC).
Os dias de permanência são a soma de DIAS_PERM, a mesma medida da caracterização da
base (Tabela 10), e não os pacientes-dia distribuídos pela Etapa 3.
"""

from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq


RAIZ    = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "pipeline" / "saidas" / "etapa1_base_limpa.parquet"
SAIDA   = RAIZ / "tests" / "saidas" / "relatorio_recorte_geografico.txt"

UFS = {"11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
       "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
       "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP", "41": "PR",
       "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF"}

_linhas: list[str] = []


def log(texto="") -> None:
    print(texto)
    _linhas.append(str(texto))


def milhar(v) -> str:
    return f"{v:,.0f}".replace(",", ".")


def carregar() -> pd.DataFrame:
    """Agrega registros e DIAS_PERM por UF de internação, UF de residência e ano de saída."""
    partes = []
    arq = pq.ParquetFile(ENTRADA)
    for lote in arq.iter_batches(columns=["MUNIC_MOV", "MUNIC_RES", "DIAS_PERM", "DT_SAIDA"],
                                 batch_size=500_000):
        d = pd.DataFrame({
            "uf_mov": pc.utf8_slice_codeunits(lote["MUNIC_MOV"].cast("string"), 0, 2),
            "uf_res": pc.utf8_slice_codeunits(lote["MUNIC_RES"].cast("string"), 0, 2),
            "dias": lote["DIAS_PERM"].to_numpy().astype("int64"),
            "ano": pc.year(lote["DT_SAIDA"]).to_numpy(),
        })
        partes.append(d.groupby(["uf_mov", "uf_res", "ano"], dropna=False)
                       .agg(registros=("dias", "size"), dias=("dias", "sum")))
    return pd.concat(partes).groupby(level=[0, 1, 2], dropna=False).sum().reset_index()


def main() -> None:
    g = carregar()
    tot_reg, tot_dias = g["registros"].sum(), g["dias"].sum()

    log("Recorte geográfico da base tratada (etapa1_base_limpa.parquet)")
    log(f"Registros: {milhar(tot_reg)}; soma de DIAS_PERM: {milhar(tot_dias)}\n")

    log("UF de internação (MUNIC_MOV)")
    mov = g.groupby("uf_mov")[["registros", "dias"]].sum()
    for uf, r in mov.iterrows():
        log(f"  {UFS.get(uf, uf):<4} registros {milhar(r.registros):>10}  dias {milhar(r.dias):>11}")

    log("\nUF de residência (MUNIC_RES)")
    res = g.groupby("uf_res")[["registros", "dias"]].sum().sort_values("registros", ascending=False)
    for uf, r in res.iterrows():
        log(f"  {UFS.get(uf, uf):<4} registros {milhar(r.registros):>10} "
            f"({100 * r.registros / tot_reg:6.2f}%)  dias {milhar(r.dias):>11} "
            f"({100 * r.dias / tot_dias:6.2f}%)")

    fora = g[g["uf_res"] != "42"]
    log(f"\nResidentes fora de SC: {milhar(fora['registros'].sum())} registros "
        f"({100 * fora['registros'].sum() / tot_reg:.2f}%) e {milhar(fora['dias'].sum())} "
        f"dias de permanência ({100 * fora['dias'].sum() / tot_dias:.2f}%)")

    log("\nParticipação dos residentes fora de SC por ano de saída (% dos dias de permanência)")
    ano = g.assign(fora=g["uf_res"] != "42").groupby(["ano", "fora"])["dias"].sum().unstack(fill_value=0)
    for a, r in ano.iterrows():
        if a >= 2015:
            log(f"  {a}  {100 * r[True] / (r[True] + r[False]):5.2f}%")

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text("\n".join(_linhas) + "\n", encoding="utf-8")
    print(f"\nRelatório gravado em {SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
