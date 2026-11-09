import pyarrow.parquet as pq

arquivo = "pipeline/saidas/etapa1_base_limpa.parquet"

print(pq.ParquetFile(arquivo).schema_arrow)