import urllib.request

#url = "ftp://ftp.datasus.gov.br/dissemin/publicos/SIHSUS/200801_/Doc/IT_SIHSUS_1603.pdf"

url= "ftp://ftp.datasus.gov.br/dissemin/publicos/SIHSUS/200801_/Dados/RDSC2601.dbc"
destino = "documentacao_sihsus.pdf"

print("Iniciando download...")
urllib.request.urlretrieve(url, destino)
print(f"Arquivo salvo como: {destino}")