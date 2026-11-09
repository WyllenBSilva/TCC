import os
import subprocess

# Configurações iniciais
diretorio_destino = "data_dbc"
#diretorio_destino = "data_CNES"
anos = range(2015, 2026)  # De 2020 a 2025
meses = [f"{m:02d}" for m in range(1, 13)]  # ['01', '02', ..., '12']
estados = ["SC"]  # Adicione outros estados aqui se precisar (ex: ["SP", "RJ", "SC"])

# Cria o diretório se ele não existir
os.makedirs(diretorio_destino, exist_ok=True)

print(f"Iniciando downloads em: {os.path.abspath(diretorio_destino)}")

for ano in anos:
    ano_dois_digitos = str(ano)[2:]  # Ex: 2025 -> '25'
    
    for mes in meses:
        for estado in estados:
            # Nome do arquivo. Ex: RDSC2507.dbc
            nome_arquivo = f"RD{estado}{ano_dois_digitos}{mes}.dbc"
            #nome_arquivo = f"LT{estado}{ano_dois_digitos}{mes}.dbc"
            
            # URL completa do FTP
            url = f"ftp://ftp.datasus.gov.br/dissemin/publicos/SIHSUS/200801_/Dados/{nome_arquivo}"
            #url = f"ftp://ftp.datasus.gov.br/dissemin/publicos/CNES/200508_/Dados/LT/{nome_arquivo}"
            print(f"Baixando: {nome_arquivo}...")
            
            # Comando Wget
            # -P define a pasta de destino
            # -q torna o download "quiet" (opcional, tire se quiser ver o progresso do wget)
            
            
            #comando = ["wget", "-P", diretorio_destino, "", url]
            comando = ["wget", "-q", "-nc", "-P", diretorio_destino, url]
            try:
                # Executa o comando no sistema
                subprocess.run(comando, check=True)
            except subprocess.CalledProcessError:
                # O FTP do DATASUS às vezes não tem um mês específico ou cai. 
                # Isso evita que o script trave caso o arquivo não exista.
                print(f" [Aviso] Não foi possível baixar ou não existe: {nome_arquivo}")

print("\nProcesso concluído!")