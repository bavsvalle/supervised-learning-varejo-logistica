"""
Junta a amostra de 60 mil estabelecimentos (com o alvo `risco` já definido)
com as variáveis explicativas vindas de Empresas e Simples/MEI, casando
pelo cnpj_basico. Gera o dataset final pronto para a etapa de modelagem.
"""
import pandas as pd

CAMINHO_AMOSTRA = "data/amostra_varejo_60k.parquet"
CAMINHO_EMPRESAS = "data/raw/empresa/cohort_empresas.parquet"
CAMINHO_SIMPLES = "data/raw/simples/cohort_simples.parquet"
CAMINHO_SAIDA = "data/dataset_modelagem.parquet"

amostra = pd.read_parquet(CAMINHO_AMOSTRA)
empresas = pd.read_parquet(CAMINHO_EMPRESAS)
simples = pd.read_parquet(CAMINHO_SIMPLES)

print(f"Amostra: {amostra.shape}")
print(f"Empresas: {empresas.shape}")
print(f"Simples: {simples.shape}")

# left join a partir da amostra: toda empresa da amostra tem que permanecer no dataset final,
# mesmo que não tenha casado com Empresas/Simples (não deveria acontecer, mas left evita perder linha)
dataset = amostra.merge(empresas, on="cnpj_basico", how="left")
dataset = dataset.merge(simples, on="cnpj_basico", how="left")

# Ausência em Simples/MEI significa "nunca optou", não dado faltante --
# por isso vira False em vez de nulo (ver explicação anterior sobre
# cobertura de 97,7% em vez de 100%).
dataset["optante_simples"] = dataset["optante_simples"].fillna(False)
dataset["optante_mei"] = dataset["optante_mei"].fillna(False)

print(f"\nDataset final: {dataset.shape}")
print("\nValores nulos por coluna:")
print(dataset.isna().sum()[dataset.isna().sum() > 0])

dataset.to_parquet(CAMINHO_SAIDA, index=False)
print(f"\nSalvo em {CAMINHO_SAIDA}")