"""
diagnostico_consolidado.py
==========================
Script para carregar a base consolidada em Parquet e gerar
um relatório de diagnóstico dos dados brutos, incluindo
análise de registros duplicados.
"""

from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURAÇÕES
# ============================================================
ARQUIVO_ALVO = Path("data_all_concatenado/dados_concatenados.parquet")
DIAS_PERM_LIMITE_SUPERIOR = 365

# ============================================================
# FUNÇÃO DE DIAGNÓSTICO
# ============================================================
def gerar_relatorio_diagnostico(df: pd.DataFrame, nome_arquivo: str) -> None:
    """
    Imprime um relatório de qualidade dos dados focando no arquivo
    concatenado, analisando nulidades, anomalias temporais e duplicados.
    """
    print(f"\n{'='*60}")
    print(f"DIAGNÓSTICO: {nome_arquivo}")
    print(f"{'='*60}")
    
    n_registros = len(df)
    print(f"  Registros totais   : {n_registros:>10,}")
    print(f"  Colunas            : {len(df.columns):>10}")

    # --------------------------------------------------------
    # ANÁLISE DE DUPLICADOS
    # --------------------------------------------------------
    print(f"\n  [DUPLICIDADE]")
    
    # 1. Linhas 100% idênticas
    duplicados_totais = df.duplicated().sum()
    print(f"  Linhas idênticas   : {duplicados_totais:>10,} ({duplicados_totais/n_registros:.2%})")
    
    # 2. AIHs repetidas (se a coluna existir)
    if "N_AIH" in df.columns:
        aihs_duplicadas = df.duplicated(subset=["N_AIH"]).sum()
        print(f"  N_AIH repetidos    : {aihs_duplicadas:>10,} ({aihs_duplicadas/n_registros:.2%})")
        if aihs_duplicadas > 0:
            print("  * Nota: N_AIH repetido pode ser erro ou AIH de longa permanência (IDENT=5).")

    # --------------------------------------------------------
    # ANÁLISE DE PERMANÊNCIA (DIAS_PERM)
    # --------------------------------------------------------
    if "DIAS_PERM" in df.columns:
        zeros   = (df["DIAS_PERM"] == 0).sum()
        neg     = (df["DIAS_PERM"] < 0).sum()
        grandes = (df["DIAS_PERM"] > DIAS_PERM_LIMITE_SUPERIOR).sum()
        
        print(f"\n  [TEMPO DE PERMANÊNCIA]")
        print(f"  DIAS_PERM = 0      : {zeros:>10,} ({zeros/n_registros:.2%})")
        print(f"  DIAS_PERM < 0      : {neg:>10,}")
        print(f"  DIAS_PERM > {DIAS_PERM_LIMITE_SUPERIOR:<3}  : {grandes:>10,}")
        print(f"  DIAS_PERM máximo   : {df['DIAS_PERM'].max():>10}")

    # --------------------------------------------------------
    # IDENTIFICADOR DE LONGA PERMANÊNCIA
    # --------------------------------------------------------
    if "IDENT" in df.columns:
        longa = (df["IDENT"].astype(str) == "5").sum()
        print(f"\n  [TIPO DE AIH]")
        print(f"  IDENT=5 (longa perm): {longa:>9,} ({longa/n_registros:.2%})")

    # --------------------------------------------------------
    # DADOS DEMOGRÁFICOS
    # --------------------------------------------------------
    if "SEXO" in df.columns:
        print(f"\n  [SEXO - Distribuição]")
        print(df['SEXO'].value_counts(dropna=False).to_string())

    if "COD_IDADE" in df.columns:
        print(f"\n  [COD_IDADE - Distribuição]")
        print(df['COD_IDADE'].value_counts(dropna=False).to_string())

    # --------------------------------------------------------
    # VERIFICAÇÃO DE VALORES NULOS EM COLUNAS CRÍTICAS
    # --------------------------------------------------------
    print(f"\n  [VALORES NULOS EM COLUNAS CRÍTICAS]")
    for col in ["DT_INTER", "DT_SAIDA", "DIAG_PRINC", "N_AIH"]:
        if col in df.columns:
            nulos = df[col].isna().sum()
            print(f"  {col:<18} : {nulos:>10,}")

    print("\n" + "="*60 + "\n")

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================
if __name__ == "__main__":
    if not ARQUIVO_ALVO.exists():
        print(f"Erro: O arquivo '{ARQUIVO_ALVO}' não foi encontrado.")
        print("Certifique-se de que o script de concatenação foi executado antes.")
    else:
        print(f"Carregando a base consolidada na memória: {ARQUIVO_ALVO}...")
        df_consolidado = pd.read_parquet(ARQUIVO_ALVO)
        gerar_relatorio_diagnostico(df_consolidado, ARQUIVO_ALVO.name)