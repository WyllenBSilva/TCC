from pathlib import Path
import pandas as pd

# Diretórios
diretorio_entrada = Path("data_parquet")
diretorio_saida = Path("data_parquet_colunas_reduzidas")

# Cria o diretório de saída caso não exista
diretorio_saida.mkdir(parents=True, exist_ok=True)

# Colunas selecionadas para o trabalho
colunas_desejadas = [
    "N_AIH",
    "DT_INTER",
    "ANO_CMPT",
    "MES_CMPT",
    "DT_SAIDA",
    "DIAS_PERM",
    "NASC",
    "IDADE",
    "COD_IDADE",
    "SEXO",
    "ESPEC",
    "DIAG_PRINC",
    "IDENT",  
    #"DIAGSEC1",    -- Não serão incluídos por terem muitos registros nulos.
    "MUNIC_RES",
    "MUNIC_MOV",
    "MORTE",
    "CAR_INT",
    "RACA_COR",
]

# Lista os arquivos Parquet
arquivos = sorted(diretorio_entrada.glob("*.parquet"))

if not arquivos:
    print(f"Nenhum arquivo .parquet encontrado em: {diretorio_entrada}")
else:
    print(f"{len(arquivos)} arquivos encontrados.\n")

    for arquivo in arquivos:
        try:
            print(f"Processando: {arquivo.name}")

            # Lê o arquivo
            df = pd.read_parquet(arquivo)

            # Verifica quais colunas existem no arquivo
            colunas_existentes = [
                coluna for coluna in colunas_desejadas
                if coluna in df.columns
            ]

            # Identifica colunas ausentes
            colunas_ausentes = [
                coluna for coluna in colunas_desejadas
                if coluna not in df.columns
            ]

            # Mantém somente as colunas disponíveis
            df_reduzido = df[colunas_existentes].copy()

            # Caminho do arquivo de saída
            caminho_saida = diretorio_saida / arquivo.name

            # Salva o arquivo reduzido
            df_reduzido.to_parquet(
                caminho_saida,
                index=False
            )

            print(
                f"  ✓ Salvo: {caminho_saida.name} "
                f"({len(df_reduzido):,} registros, "
                f"{len(colunas_existentes)} colunas)"
            )

            if colunas_ausentes:
                print(
                    f"  ⚠ Colunas ausentes: "
                    f"{', '.join(colunas_ausentes)}"
                )

            print()

        except Exception as erro:
            print(f"  ✗ Erro ao processar {arquivo.name}: {erro}\n")

    print("Processamento concluído!")