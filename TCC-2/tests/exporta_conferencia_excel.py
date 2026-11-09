#!/usr/bin/env python3
"""
exporta_conferencia_excel.py
============================
Exporta os dados para conferir no Excel a validação do corte da janela.
A base inteira não cabe em uma planilha (limite de 1.048.576 linhas), então são
gravados só os registros que ocupam leito em um mês escolhido.

Uso
---
    python tests/exporta_conferencia_excel.py

Entrada (somente leitura):
    pipeline/saidas/etapa1_base_limpa.parquet
    tests/saidas/completude_por_mes_e_defasagem.csv

Saídas (em tests/saidas/excel/), com separador ";" e datas dd/mm/aaaa:
    registros_2024-08.csv   registros com ocupação em agosto de 2024
    registros_2025-08.csv   registros com ocupação em agosto de 2025
    completude_por_mes.csv  pacientes-dia finais e conhecidos com k competências posteriores

Registros com ocupação no mês M:
    IDENT != 5: DT_INTER antes do fim de M e DT_SAIDA depois do início de M;
    IDENT  = 5: DT_SAIDA dentro de M.
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


RAIZ    = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "pipeline" / "saidas" / "etapa1_base_limpa.parquet"
ARQ_CSV = RAIZ / "tests" / "saidas" / "completude_por_mes_e_defasagem.csv"
SAIDA   = RAIZ / "tests" / "saidas" / "excel"

TAM_LOTE = 250_000
COLUNAS  = ["N_AIH", "IDENT", "DT_INTER", "DT_SAIDA", "DIAS_PERM", "ANO_CMPT", "MES_CMPT"]
MESES    = ["2024-08", "2025-08"]


def registros_do_mes(mes: str) -> pd.DataFrame:
    ini = pd.Timestamp(f"{mes}-01")
    fim = ini + pd.offsets.MonthBegin(1)
    partes = []
    for lote in pq.ParquetFile(ENTRADA).iter_batches(batch_size=TAM_LOTE, columns=COLUNAS):
        d = lote.to_pandas()
        is5 = d["IDENT"] == 5
        comum = ~is5 & (d["DT_INTER"] < fim) & (d["DT_SAIDA"] > ini)
        fatia = is5 & (d["DT_SAIDA"] >= ini) & (d["DT_SAIDA"] < fim)
        partes.append(d[comum | fatia])
    return pd.concat(partes, ignore_index=True)


def conferir(d: pd.DataFrame, mes: str, ref: pd.DataFrame) -> None:
    """Repete em pandas a conta da planilha e compara com a validação."""
    ini = pd.Timestamp(f"{mes}-01")
    fim = ini + pd.offsets.MonthBegin(1)
    dias = (d["DT_SAIDA"].clip(upper=fim) - d["DT_INTER"].clip(lower=ini)).dt.days
    dias = dias.where(d["IDENT"] != 5, d["DIAS_PERM"])
    print(f"{mes}: {len(d):,} registros | soma {int(dias.sum()):,} "
          f"(validação: {int(ref.loc[mes, 'pacientes_dia']):,})")
    limite = ini.year * 12 + ini.month - 1 + 4
    ate_k4 = dias[d["ANO_CMPT"].astype(int) * 12 + d["MES_CMPT"].astype(int) - 1 <= limite].sum()
    print(f"  conhecido com 4 competências posteriores: {int(ate_k4):,} "
          f"(validação: {int(ref.loc[mes, 'ate_k4']):,})")


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(ARQ_CSV, index_col="ano_mes")
    ref.to_csv(SAIDA / "completude_por_mes.csv", sep=";")

    for mes in MESES:
        d = registros_do_mes(mes)
        conferir(d, mes, ref)
        d.to_csv(SAIDA / f"registros_{mes}.csv", sep=";", index=False, date_format="%d/%m/%Y")


if __name__ == "__main__":
    main()
