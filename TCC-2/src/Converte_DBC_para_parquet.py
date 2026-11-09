import os
from pathlib import Path
import pandas as pd
import pyreaddbc
from dbfread import DBF

def extrair_data_ordenacao(caminho_arquivo):
    """
    Função auxiliar para extrair o Ano e Mês do nome do arquivo DATASUS (ex: RDSC1501).
    Retorna uma tupla (Ano, Mês) para o Python ordenar corretamente.
    """
    stem = caminho_arquivo.stem.upper()
    try:
        # Pega os 4 últimos caracteres do nome (ex: "1501" de "RDSC1501")
        yy = int(stem[-4:-2])  # "15" -> 15
        mm = int(stem[-2:])    # "01" -> 1
        
        # Converte o ano de 2 dígitos (ex: 15 -> 2015, 98 -> 1998)
        ano = 2000 + yy if yy < 80 else 1900 + yy
        
        return (ano, mm, stem)
    except (ValueError, IndexError):
        # Se algum arquivo não seguir o padrão de nome do DATASUS, joga pro final da lista
        return (9999, 99, stem)

def converter_dbc_para_parquet():
    pasta_entrada = Path("data_dbc")
    pasta_saida = Path("data_parquet")
    
    pasta_saida.mkdir(parents=True, exist_ok=True)

    if not pasta_entrada.exists() or not pasta_entrada.is_dir():
        print(f"Atenção: O diretório '{pasta_entrada}' não foi encontrado.")
        return

    # Busca os arquivos e remove possíveis duplicados
    arquivos_dbc = list(pasta_entrada.glob("*.dbc")) + list(pasta_entrada.glob("*.DBC"))
    arquivos_dbc = list(set(arquivos_dbc))

    if not arquivos_dbc:
        print("Nenhum arquivo .dbc encontrado.")
        return

    # --- ALTERAÇÃO AQUI: Ordena a lista usando a função de data ---
    arquivos_dbc.sort(key=extrair_data_ordenacao)

    print(f"Total de arquivos encontrados: {len(arquivos_dbc)}")
    print(f"Primeiro a processar: {arquivos_dbc[0].name}")
    print(f"Último a processar: {arquivos_dbc[-1].name}")
    print("\nIniciando conversão em ordem cronológica...\n")

    for caminho_dbc in arquivos_dbc:
        nome_arquivo_saida = f"{caminho_dbc.stem}.parquet"
        caminho_parquet = pasta_saida / nome_arquivo_saida
        
        caminho_dbf_temp = pasta_saida / f"{caminho_dbc.stem}_temp.dbf"

        print(f"Processando: {caminho_dbc.name}...", end=" ")

        try:
            # 1. Descompacta DBC para DBF temporário
            pyreaddbc.dbc2dbf(str(caminho_dbc), str(caminho_dbf_temp))
            
            # 2. Lê o DBF gerado
            tabela_dbf = DBF(str(caminho_dbf_temp), encoding="iso-8859-1")
            
            # 3. Transforma em DataFrame Pandas
            df = pd.DataFrame(iter(tabela_dbf))
            
            # 4. Salva em Parquet (Sobrescreve se o arquivo já existir)
            df.to_parquet(caminho_parquet, index=False, engine="pyarrow")
            
            # 5. Apaga o DBF temporário
            if caminho_dbf_temp.exists():
                os.remove(caminho_dbf_temp)
                
            print("[OK]")
        except Exception as e:
            print(f"[ERRO] Detalhes: {e}")

if __name__ == "__main__":
    converter_dbc_para_parquet()