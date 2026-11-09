import os
import glob
import pandas as pd
from dbfread import DBF

diretorio_dbf = "data_convert/dados_convertidos_RD"
diretorio_parquet = "dados_parquet"
os.makedirs(diretorio_parquet, exist_ok=True)

arquivos_dbf = glob.glob(os.path.join(diretorio_dbf, "*.dbf"))

print(f"Encontrados {len(arquivos_dbf)} arquivos .dbf para converter.")

for caminho_dbf in arquivos_dbf:
    nome_base = os.path.splitext(os.path.basename(caminho_dbf))[0]
    caminho_parquet = os.path.join(diretorio_parquet, f"{nome_base}.parquet")
    
    print(f"Convertendo {nome_base} para Parquet...")
    try:
        # Carrega o DBF especificando codificação padrão do DATASUS
        tabela = DBF(caminho_dbf, encoding='iso-8859-1')
        df = pd.DataFrame(iter(tabela))
        
        # Salva em formato colunar de alta performance
        df.to_parquet(caminho_parquet, index=False, engine='pyarrow')
        print(f" -> [Sucesso]: {caminho_parquet}")
    except Exception as e:
        print(f" -> [Erro] Falha ao processar {nome_base}: {e}")

print("\nConversão concluída!")