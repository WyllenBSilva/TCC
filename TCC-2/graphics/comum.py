#!/usr/bin/env python3
"""
comum.py
========
Caminhos, leitura em lotes e estilo compartilhados pelos gráficos.

Nos gráficos de caracterização, a unidade é o REGISTRO do arquivo RD, não o paciente:
a base não tem identificador de paciente, e uma internação longa gera um registro por
mês. Nos gráficos de resultados, a unidade é o mês e o valor é pacientes-dia.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq


RAIZ    = Path(__file__).resolve().parent.parent
PIPELINE = RAIZ / "pipeline" / "saidas"
ENTRADA = PIPELINE / "etapa1_base_limpa.parquet"
FIGURAS = RAIZ / "monografia" / "figuras"
TABELAS = Path(__file__).resolve().parent / "saidas"

TAM_LOTE = 250_000

AZUL    = "#2a78d6"
LARANJA = "#eb6834"
TINTA   = "#1a1a1a"
GRADE   = "#d9d9d9"
CINZA   = "#767676"
FUNDO   = "#f0f0f0"


def eixo_limpo(ax, grade="y") -> None:
    """Grade recessiva, sem marcas de escala e sem as bordas de cima e da direita."""
    ax.grid(axis=grade, color=GRADE, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def lotes(colunas: list[str]):
    """Devolve a base limpa em lotes de pandas, só com as colunas pedidas."""
    for lote in pq.ParquetFile(ENTRADA).iter_batches(batch_size=TAM_LOTE, columns=colunas):
        yield lote.to_pandas()


def total_registros() -> int:
    return pq.ParquetFile(ENTRADA).metadata.num_rows


def milhar(valor: float) -> str:
    """Inteiro com separador de milhar brasileiro: 1234567 vira 1.234.567."""
    return f"{valor:,.0f}".replace(",", ".")


def percentual(valor: float) -> str:
    """Percentual com duas casas e vírgula decimal: 55.72 vira 55,72%."""
    return f"{valor:.2f}".replace(".", ",") + "%"


def aplicar_estilo() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "text.color": TINTA,
        "axes.labelcolor": TINTA,
        "axes.edgecolor": TINTA,
        "xtick.color": TINTA,
        "ytick.color": TINTA,
        "hatch.linewidth": 1.0,
    })


def salvar(fig, nome: str) -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    destino = FIGURAS / nome
    fig.savefig(destino, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figura gravada: {destino.relative_to(RAIZ)}")
