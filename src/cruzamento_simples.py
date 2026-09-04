"""
Cruza o arquivo bruto de Simples/MEI (Receita Federal) com a coorte de
varejo já filtrada, usando DuckDB (mesma abordagem validada para Empresas).

Layout oficial do arquivo SIMPLES.CSV (7 colunas, sem cabeçalho, ; como
separador, latin-1):
  CNPJ_BASICO; OPCAO_PELO_SIMPLES; DATA_OPCAO_SIMPLES; DATA_EXCLUSAO_SIMPLES;
  OPCAO_PELO_MEI; DATA_OPCAO_MEI; DATA_EXCLUSAO_MEI
"""
import glob
import time
import duckdb

CAMINHO_COORTE = "data/cohort_varejo_completo.parquet"
PASTA_SIMPLES_BRUTO = "data/raw/simples"
CAMINHO_SAIDA = "data/raw/simples/cohort_simples.parquet"

inicio = time.time()
con = duckdb.connect()

# mesma lógica de cruzamento_empresas.py: filtra o CSV bruto de Simples/MEI direto na leitura,
# usando só os CNPJs da coorte de varejo
con.execute(f"""
    CREATE TEMP TABLE coorte AS
    SELECT DISTINCT cnpj_basico FROM read_parquet('{CAMINHO_COORTE}')
""")
total_coorte = con.execute("SELECT COUNT(*) FROM coorte").fetchone()[0]
print(f"CNPJs básicos na coorte de varejo: {total_coorte:,}")

# Arquivo único (não vem dividido em 10 partes como Empresas/Estabelecimentos)
arquivos = glob.glob(f"{PASTA_SIMPLES_BRUTO}/*SIMPLES*")
print(f"Arquivo(s) encontrado(s): {arquivos}")
caminho_csv = arquivos[0]

query = f"""
    COPY (
        SELECT
            e.c0 AS cnpj_basico,
            -- optante_pelo_simples/mei vem como 'S'/'N' no arquivo bruto; converte direto para booleano
            (e.c1 = 'S') AS optante_simples,
            e.c2 AS data_opcao_simples,
            e.c3 AS data_exclusao_simples,
            (e.c4 = 'S') AS optante_mei,
            e.c5 AS data_opcao_mei,
            e.c6 AS data_exclusao_mei
        FROM read_csv('{caminho_csv}', delim=';', header=false,
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
print(f"\nRegistros de Simples casados com a coorte: {total_saida:,}")
print(f"Cobertura: {total_saida / total_coorte:.1%} da coorte encontrada em Simples")
print(f"Salvo em {CAMINHO_SAIDA}")
print(f"Tempo total: {time.time() - inicio:.1f}s")