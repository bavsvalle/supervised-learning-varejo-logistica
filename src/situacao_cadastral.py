import pandas as pd
import glob

# Segunda etapa do pipeline: junta os 10 arquivos parciais de cohort (um por zip de Estabelecimentos
# processado por pipeline_extracao.py) num único dataset consolidado de varejo 2017-2019
arquivos = sorted(glob.glob("data/raw/cohort_estabelecimentos*.parquet"))
df_completo = pd.concat([pd.read_parquet(f) for f in arquivos], ignore_index=True)
print(f"Total consolidado: {len(df_completo)} linhas")
# checagem rápida da distribuição de situacao_cadastral antes de salvar -- serve de base para as
# decisões de amostragem/alvo feitas em preparacao_amostra.py e redefinir_alvo.py
print(df_completo["situacao_cadastral"].value_counts())
df_completo.to_parquet("data/cohort_varejo_completo.parquet", index=False)