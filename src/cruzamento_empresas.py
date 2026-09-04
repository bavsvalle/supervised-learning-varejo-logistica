"""
Cruza os arquivos brutos de Empresas (Receita Federal) com a coorte de
varejo já filtrada, usando DuckDB em vez de pandas puro — leitura
vetorizada do CSV, muito mais rápida para dezenas de milhões de linhas.

Layout oficial do arquivo EMPRECSV (7 colunas, sem cabeçalho, ; como
separador, latin-1, decimal com vírgula):
  CNPJ_BASICO; RAZAO_SOCIAL; NATUREZA_JURIDICA; QUALIFICACAO_RESPONSAVEL;
  CAPITAL_SOCIAL; PORTE_EMPRESA; ENTE_FEDERATIVO_RESPONSAVEL
"""
import time
import duckdb
import pandas as pd

CAMINHO_COORTE = "data/cohort_varejo_completo.parquet"
PASTA_EMPRESAS_BRUTO = "data/raw/empresa"
CAMINHO_SAIDA = "data/raw/empresa/cohort_empresas.parquet"

inicio = time.time()
con = duckdb.connect()

# tabela temporária só com os CNPJs básicos da coorte -- o join usa essa lista pequena para
# filtrar o arquivo bruto de Empresas (dezenas de milhões de linhas) direto na leitura via DuckDB
con.execute(f"""
    CREATE TEMP TABLE coorte AS
    SELECT DISTINCT cnpj_basico FROM read_parquet('{CAMINHO_COORTE}')
""")
total_coorte = con.execute("SELECT COUNT(*) FROM coorte").fetchone()[0]
print(f"CNPJs básicos na coorte de varejo: {total_coorte:,}")

query = f"""
    COPY (
        SELECT
            e.c0 AS cnpj_basico,
            e.c1 AS razao_social,
            e.c2 AS natureza_juridica,
            e.c3 AS qualificacao_responsavel,
            -- capital_social vem com vírgula decimal (padrão BR) no CSV bruto; troca por ponto
            -- antes do cast para double, senão TRY_CAST falha silenciosamente e vira nulo
            TRY_CAST(REPLACE(e.c4, ',', '.') AS DOUBLE) AS capital_social,
            e.c5 AS porte_empresa,
            e.c6 AS ente_federativo_responsavel
        FROM read_csv('{PASTA_EMPRESAS_BRUTO}/*.EMPRECSV', delim=';', header=false,
            encoding='latin-1', quote='"', escape='"', strict_mode=false,
            columns={{
                'c0': 'VARCHAR', 'c1': 'VARCHAR', 'c2': 'VARCHAR', 'c3': 'VARCHAR',
                'c4': 'VARCHAR', 'c5': 'VARCHAR', 'c6': 'VARCHAR'
            }}) AS e
        INNER JOIN coorte c ON e.c0 = c.cnpj_basico
    ) TO '{CAMINHO_SAIDA}' (FORMAT PARQUET)
"""
con.execute(query)

total_saida = con.execute(f"SELECT COUNT(*) FROM read_parquet('{CAMINHO_SAIDA}')").fetchone()[0]
print(f"\nRegistros de Empresas casados com a coorte: {total_saida:,}")
print(f"Cobertura: {total_saida / total_coorte:.1%} da coorte encontrada em Empresas")
print(f"Salvo em {CAMINHO_SAIDA}")
print(f"Tempo total: {time.time() - inicio:.1f}s")