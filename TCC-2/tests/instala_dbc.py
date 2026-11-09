import urllib.request
from pathlib import Path

# Pasta destino no Linux
caminho_pasta = Path('/home/wylle/udesc/TCC/dados')

# Cria a pasta caso não exista
caminho_pasta.mkdir(parents=True, exist_ok=True)

# URL base do DATASUS
base_url = "ftp://ftp.datasus.gov.br/dissemin/publicos/SIHSUS/200801_/Dados"

# Ano 2025 -> "25"
ano = "25"

# UF
uf = "SC"

print("Iniciando downloads...\n")

# Meses de 01 até 12
for mes in range(1, 13):

    mes_str = f"{mes:02d}"

    # Nome do arquivo
    arquivo = f"RD{uf}{ano}{mes_str}.dbc"

    # URL completa
    url = f"{base_url}/{arquivo}"
    print(url)
    # Caminho final do arquivo
    destino = caminho_pasta / arquivo

    # Evita baixar novamente
    if destino.exists():
        print(f"Arquivo já existe: {arquivo}")
        continue

    print(f"Baixando: {arquivo}")

    try:
        urllib.request.urlretrieve(url, destino)

        print(f"Salvo em: {destino}")

    except Exception as e:
        print(f"Erro ao baixar {arquivo}: {e}")

print("\nTodos os downloads finalizaram.")