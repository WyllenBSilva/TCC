"""
concatenar_parquets.py
======================
Script para ler e concatenar todos os arquivos Parquet de um diretório
e salvar o resultado consolidado em um novo diretório.
"""

from pathlib import Path
import pandas as pd


def concatenar_arquivos_parquet(
    diretorio_entrada: Path,
    diretorio_saida: Path,
    nome_arquivo_saida: str = "dados_concatenados.parquet"
) -> None:
    # Garantir que os caminhos sejam objetos Path
    diretorio_entrada = Path(diretorio_entrada)
    diretorio_saida = Path(diretorio_saida)

    # Buscar todos os arquivos .parquet no diretório de entrada
    arquivos = sorted(diretorio_entrada.glob("*.parquet"))

    if not arquivos:
        print(f"Nenhum arquivo .parquet encontrado em: {diretorio_entrada}")
        return

    print(f"Encontrados {len(arquivos)} arquivos para concatenar...\n")

    lista_dfs = []
    total_linhas_individuais = 0

    # Leitura sequencial dos arquivos
    for idx, arquivo in enumerate(arquivos, start=1):
        print(f"[{idx}/{len(arquivos)}] Lendo: {arquivo.name}...")
        df = pd.read_parquet(arquivo)
        lista_dfs.append(df)
        total_linhas_individuais += len(df)

    # Concatenar todos os DataFrames da lista
    print("\nConcatenando DataFrames...")
    df_final = pd.concat(lista_dfs, ignore_index=True)

    # Criar o diretório de saída caso não exista
    diretorio_saida.mkdir(parents=True, exist_ok=True)

    # Caminho final do arquivo
    caminho_saida = diretorio_saida / nome_arquivo_saida

    print(f"Salvando arquivo consolidado em: {caminho_saida}...")
    df_final.to_parquet(caminho_saida, index=False)

    print("\n" + "=" * 60)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f"  • Total de arquivos processados : {len(arquivos):,}")
    print(f"  • Total de registros salvos     : {len(df_final):,}")
    print(f"  • Total de colunas              : {len(df_final.columns)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    DIRETORIO_ENTRADA = Path("data_parquet_colunas_reduzidas")
    DIRETORIO_SAIDA = Path("data_all_concatenado")

    concatenar_arquivos_parquet(DIRETORIO_ENTRADA, DIRETORIO_SAIDA)