"""
limpeza_dados.py
================
Limpeza e tratamento de inconsistências dos arquivos RD do SIH-SUS.

Etapas implementadas
--------------------
1.  Diagnóstico inicial do arquivo (relatório antes de qualquer alteração)
2.  Conversão de datas (DT_INTER, DT_SAIDA, NASC) de string para datetime
3.  Remoção de registros com datas obrigatórias ausentes ou malformadas
4.  Detecção e tratamento de registros onde DT_SAIDA < DT_INTER
5.  Remoção de registros com COD_IDADE == 0 (idade inválida/tratada equivocadamente)
6.  Tratamento de DIAS_PERM = 0 (internações no mesmo dia)
7.  Conversão de IDADE + COD_IDADE para IDADE_ANOS (valor decimal em anos)
8.  Padronização de SEXO (de código numérico para rótulo legível)
9.  Extração do capítulo CID-10 a partir de DIAG_PRINC
10. Decodificação de CAR_INT (caráter da internação)
11. Sinalização de AIH de longa permanência (IDENT = 5)
12. Criação de variáveis temporais para modelagem
    (ANO, MES, TRIMESTRE, SEMESTRE, ESTACAO)
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

DIRETORIO_ENTRADA = Path("data_parquet_colunas_reduzidas")
DIRETORIO_SAIDA   = Path("dados_processados_limpos")

# Limite superior de permanência considerado válido.
DIAS_PERM_LIMITE_SUPERIOR = 365


# ============================================================
# MAPEAMENTOS
# ============================================================

MAPA_SEXO = {
    "0": "Ignorado",
    "1": "Masculino",
    "2": "Ignorado",
    "3": "Feminino",
}

MAPA_CAR_INT = {
    "01": "Eletivo",
    "02": "Urgência",
    "03": "Acidente_trabalho_típico",
    "04": "Acidente_trabalho_atípico",
    "05": "Acidente_trânsito",
    "06": "Outro_acidente",
}

MAPA_CAPITULO_CID = {
    "A": "I_Infecciosas_parasitárias",
    "B": "I_Infecciosas_parasitárias",
    "C": "II_Neoplasias",
    "D": "II_III_Neoplasias_sangue",
    "E": "IV_Endócrinas",
    "F": "V_Transtornos_mentais",
    "G": "VI_Nervoso",
    "H": "VII_VIII_Olho_ouvido",
    "I": "IX_Circulatório",
    "J": "X_Respiratório",
    "K": "XI_Digestivo",
    "L": "XII_Pele",
    "M": "XIII_Músculo_esquelético",
    "N": "XIV_Geniturinário",
    "O": "XV_Gravidez_parto",
    "P": "XVI_Perinatal",
    "Q": "XVII_Congênitas",
    "R": "XVIII_Sintomas_sinais",
    "S": "XIX_Lesões",
    "T": "XIX_Lesões",
    "V": "XX_Causas_externas",
    "W": "XX_Causas_externas",
    "X": "XX_Causas_externas",
    "Y": "XX_Causas_externas",
    "Z": "XXI_Fatores_saúde",
    "U": "XXII_Finalidades_especiais",
}

MAPA_ESTACAO = {
    1:  "Verão",
    2:  "Verão",
    3:  "Outono",
    4:  "Outono",
    5:  "Outono",
    6:  "Inverno",
    7:  "Inverno",
    8:  "Inverno",
    9:  "Primavera",
    10: "Primavera",
    11: "Primavera",
    12: "Verão",
}


# ============================================================
# 1. DIAGNÓSTICO INICIAL
# ============================================================

def gerar_relatorio_diagnostico(df: pd.DataFrame, nome_arquivo: str) -> None:
    """
    Imprime um relatório de qualidade dos dados ANTES de qualquer
    transformação.
    """
    print(f"\n{'='*60}")
    print(f"DIAGNÓSTICO: {nome_arquivo}")
    print(f"{'='*60}")
    print(f"  Registros          : {len(df):>10,}")
    print(f"  Colunas            : {len(df.columns):>10}")

    if "DIAS_PERM" in df.columns:
        n = len(df)
        zeros   = (df["DIAS_PERM"] == 0).sum()
        neg     = (df["DIAS_PERM"] < 0).sum()
        grandes = (df["DIAS_PERM"] > DIAS_PERM_LIMITE_SUPERIOR).sum()
        print(f"\n  DIAS_PERM = 0      : {zeros:>8,}  ({zeros/n:.2%})")
        print(f"  DIAS_PERM < 0      : {neg:>8,}")
        print(f"  DIAS_PERM > {DIAS_PERM_LIMITE_SUPERIOR}     : {grandes:>8,}")
        print(f"  DIAS_PERM máximo   : {df['DIAS_PERM'].max():>8}")

    if "IDENT" in df.columns:
        longa = (df["IDENT"].astype(str) == "5").sum()
        print(f"\n  IDENT=5 (longa perm): {longa:>7,}")

    if "SEXO" in df.columns:
        print(f"\n  SEXO (distribuição):\n{df['SEXO'].value_counts(dropna=False).to_string()}")

    if "COD_IDADE" in df.columns:
        print(f"\n  COD_IDADE (distribuição):\n{df['COD_IDADE'].value_counts(dropna=False).to_string()}")

    for col in ["DT_INTER", "DT_SAIDA", "DIAG_PRINC"]:
        if col in df.columns:
            nulos = df[col].isna().sum()
            print(f"\n  {col} nulos          : {nulos:>8,}")

    print()


# ============================================================
# 2. CONVERSÃO DE DATAS
# ============================================================

def converter_datas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte DT_INTER, DT_SAIDA e NASC para datetime.
    """
    for coluna in ["DT_INTER", "DT_SAIDA", "NASC"]:
        if coluna in df.columns:
            df[coluna] = pd.to_datetime(
                df[coluna].astype(str).str.strip(),
                format="%Y%m%d",
                errors="coerce",
            )
    return df


# ============================================================
# 3. REMOÇÃO DE REGISTROS COM DATAS OBRIGATÓRIAS AUSENTES
# ============================================================

def remover_datas_invalidas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove registros onde DT_INTER ou DT_SAIDA são nulos (NaT).
    """
    antes = len(df)
    df = df.dropna(subset=["DT_INTER", "DT_SAIDA"]).copy()
    removidos = antes - len(df)
    if removidos > 0:
        print(f"  [datas_invalidas] Removidos {removidos:,} registros com DT_INTER/DT_SAIDA nulos.")
    return df


# ============================================================
# 4. REGISTROS COM DT_SAIDA < DT_INTER
# ============================================================

def tratar_saida_anterior_internacao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove registros onde a data de saída é anterior à de internação.
    """
    mascara = df["DT_SAIDA"] < df["DT_INTER"]
    n_inconsistentes = mascara.sum()
    if n_inconsistentes > 0:
        print(f"  [saida_anterior] Removidos {n_inconsistentes:,} registros onde DT_SAIDA < DT_INTER.")
        df = df[~mascara].copy()
    return df


# ============================================================
# 5. REMOÇÃO DE REGISTROS COM COD_IDADE = 0
# ============================================================

def remover_cod_idade_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exclui registros onde COD_IDADE é igual a 0, considerado um erro
    ou valor inválido de tratamento de idade.
    """
    if "COD_IDADE" not in df.columns:
        return df
    
    antes = len(df)
    # Converte para numérico para garantir a comparação correta
    cod_idade_num = pd.to_numeric(df["COD_IDADE"], errors="coerce")
    mascara = cod_idade_num == 0
    
    n_removidos = mascara.sum()
    if n_removidos > 0:
        df = df[~mascara].copy()
        print(f"  [cod_idade_zero] Removidos {n_removidos:,} registros com COD_IDADE = 0.")
        
    return df


# ============================================================
# 6. TRATAMENTO DE DIAS_PERM = 0
# ============================================================

def tratar_dias_perm_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata registros com DIAS_PERM = 0 (mesmo dia).
    """
    df["DIAS_PERM_ORIG"] = df["DIAS_PERM"].copy()
    df["FLAG_MESMO_DIA"] = df["DIAS_PERM"] == 0

    n_zeros = df["FLAG_MESMO_DIA"].sum()
    if n_zeros > 0:
        print(f"  [dias_perm_zero] {n_zeros:,} registros com DIAS_PERM=0 → imputados para 1.")

    df["DIAS_PERM_ADJ"] = df["DIAS_PERM"].clip(lower=1)
    return df


# ============================================================
# 7. SINALIZAÇÃO DE DIAS_PERM ATÍPICOS (> LIMITE)
# ============================================================

def sinalizar_dias_perm_atipicos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sinaliza registros com DIAS_PERM acima do limite configurado.
    """
    df["FLAG_PERM_ATIPICA"] = df["DIAS_PERM"] > DIAS_PERM_LIMITE_SUPERIOR
    n = df["FLAG_PERM_ATIPICA"].sum()
    if n > 0:
        print(f"  [dias_perm_atipicos] {n:,} registros com DIAS_PERM > {DIAS_PERM_LIMITE_SUPERIOR} sinalizados.")
    return df


# ============================================================
# 8. CONVERSÃO DE COD_IDADE + IDADE → IDADE_ANOS
# ============================================================

def converter_idade_para_anos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte o par IDADE + COD_IDADE em IDADE_ANOS (valor decimal em anos).
    """
    idade     = pd.to_numeric(df["IDADE"],     errors="coerce")
    cod_idade = pd.to_numeric(df["COD_IDADE"], errors="coerce")

    df["IDADE_ANOS"] = pd.NA

    # Horas → anos
    mask = cod_idade == 1
    df.loc[mask, "IDADE_ANOS"] = idade[mask] / (24 * 365.25)

    # Dias → anos
    mask = cod_idade == 2
    df.loc[mask, "IDADE_ANOS"] = idade[mask] / 365.25

    # Meses → anos
    mask = cod_idade == 3
    df.loc[mask, "IDADE_ANOS"] = idade[mask] / 12.0

    # Anos direto
    mask = cod_idade == 4
    df.loc[mask, "IDADE_ANOS"] = idade[mask].astype(float)

    # Acima de 100 anos
    mask = cod_idade == 5
    df.loc[mask, "IDADE_ANOS"] = 100.0 + idade[mask].astype(float)

    df["IDADE_ANOS"] = df["IDADE_ANOS"].astype("float64")

    nulos_idade = df["IDADE_ANOS"].isna().sum()
    if nulos_idade > 0:
        print(f"  [idade_anos] {nulos_idade:,} registros com IDADE_ANOS nulo (COD_IDADE desconhecido).")

    return df


# ============================================================
# 9. PADRONIZAÇÃO DE SEXO
# ============================================================

def padronizar_sexo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte o código numérico de SEXO para rótulo legível.
    """
    df["SEXO_LABEL"] = (
        df["SEXO"].astype(str).str.strip().map(MAPA_SEXO).fillna("Ignorado")
    )
    ignorados = (df["SEXO_LABEL"] == "Ignorado").sum()
    if ignorados > 0:
        print(f"  [sexo] {ignorados:,} registros com SEXO ignorado/desconhecido.")
    return df


# ============================================================
# 10. CAPÍTULO CID-10 A PARTIR DE DIAG_PRINC
# ============================================================

def extrair_capitulo_cid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai o capítulo da CID-10 a partir da primeira letra de DIAG_PRINC.
    """
    df["CAPITULO_CID"] = (
        df["DIAG_PRINC"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str[0]
        .map(MAPA_CAPITULO_CID)
        .fillna("Desconhecido")
    )
    desconhecidos = (df["CAPITULO_CID"] == "Desconhecido").sum()
    if desconhecidos > 0:
        print(f"  [capitulo_cid] {desconhecidos:,} registros com capítulo CID não mapeado.")
    return df


# ============================================================
# 11. DECODIFICAÇÃO DE CAR_INT
# ============================================================

def decodificar_car_int(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte o código de CAR_INT para rótulo legível.
    """
    if "CAR_INT" not in df.columns:
        return df
    df["CAR_INT_LABEL"] = (
        df["CAR_INT"].astype(str).str.strip().map(MAPA_CAR_INT).fillna("Outros")
    )
    return df


# ============================================================
# 12. SINALIZAÇÃO DE AIH DE LONGA PERMANÊNCIA (IDENT = 5)
# ============================================================

def sinalizar_longa_permanencia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sinaliza registros com IDENT = 5 (longa permanência).
    """
    if "IDENT" not in df.columns:
        return df
    df["FLAG_LONGA_PERMANENCIA"] = df["IDENT"].astype(str).str.strip() == "5"
    n = df["FLAG_LONGA_PERMANENCIA"].sum()
    if n > 0:
        print(f"  [longa_permanencia] {n:,} registros IDENT=5 sinalizados.")
    return df


# ============================================================
# 13. VARIÁVEIS TEMPORAIS PARA MODELAGEM
# ============================================================

def criar_variaveis_temporais(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria variáveis temporais derivadas de DT_INTER.
    """
    dt = df["DT_INTER"]

    df["ANO_INTER"]  = dt.dt.year
    df["MES_INTER"]  = dt.dt.month
    df["TRIMESTRE"]  = dt.dt.quarter
    df["SEMESTRE"]   = (dt.dt.month > 6).astype(int) + 1
    df["ESTACAO"]    = df["MES_INTER"].map(MAPA_ESTACAO)
    df["ANO_MES"]    = dt.dt.to_period("M").astype(str)

    return df


# ============================================================
# FUNÇÃO PRINCIPAL DE LIMPEZA (ORQUESTRADOR)
# ============================================================

def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa todas as etapas de limpeza e enriquecimento do DataFrame.
    """
    df = converter_datas(df)
    df = remover_datas_invalidas(df)
    df = tratar_saida_anterior_internacao(df)
    df = remover_cod_idade_zero(df)  # Etapa adicionada
    df = tratar_dias_perm_zero(df)
    df = sinalizar_dias_perm_atipicos(df)
    df = converter_idade_para_anos(df)
    df = padronizar_sexo(df)
    df = extrair_capitulo_cid(df)
    df = decodificar_car_int(df)
    df = sinalizar_longa_permanencia(df)
    df = criar_variaveis_temporais(df)
    return df


# ============================================================
# LEITURA, PROCESSAMENTO E SALVAMENTO DOS ARQUIVOS
# ============================================================

def processar_arquivos(
    diretorio_entrada: Path,
    diretorio_saida: Path,
) -> None:
    """
    Itera sobre todos os arquivos Parquet, aplica a limpeza e salva.
    """
    diretorio_entrada = Path(diretorio_entrada)
    diretorio_saida   = Path(diretorio_saida)
    diretorio_saida.mkdir(parents=True, exist_ok=True)

    arquivos = sorted(diretorio_entrada.glob("*.parquet"))

    if not arquivos:
        print(f"Nenhum arquivo .parquet encontrado em: {diretorio_entrada}")
        return

    print(f"{len(arquivos)} arquivo(s) encontrado(s) em '{diretorio_entrada}'.\n")

    for arquivo in arquivos:
        print(f"{'─'*60}")
        print(f"Processando: {arquivo.name}")

        df = pd.read_parquet(arquivo)

        # Relatório ANTES da limpeza
        gerar_relatorio_diagnostico(df, arquivo.name)

        registros_antes = len(df)

        # Limpeza
        df = limpar_dados(df)

        registros_depois = len(df)
        removidos = registros_antes - registros_depois
        print(
            f"\n  Registros antes : {registros_antes:>10,}\n"
            f"  Registros depois: {registros_depois:>10,}\n"
            f"  Removidos       : {removidos:>10,} ({removidos/max(registros_antes,1):.2%})"
        )

        # Salvamento
        arquivo_saida = diretorio_saida / arquivo.name
        df.to_parquet(arquivo_saida, index=False)
        print(f"  Salvo em: {arquivo_saida}")

    print(f"\n{'='*60}")
    print("Processamento concluído.")
    print(f"{'='*60}\n")


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    processar_arquivos(DIRETORIO_ENTRADA, DIRETORIO_SAIDA)