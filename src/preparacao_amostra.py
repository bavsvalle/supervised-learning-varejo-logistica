"""
preparar_amostra.py

Etapa 2 do pipeline: a partir do cohort completo de comércio varejista
(2017-2019) já extraído e filtrado pelo pipeline_extracao.py, este script:

1. Remove situações cadastrais fora do escopo da variável resposta (Nula, Suspensa)
2. Constrói a variável binária "risco" (0 = ativa, 1 = baixada/inapta)
3. Define e justifica o tamanho da amostra estratificada via regra EPV
4. Salva a amostra final para a etapa de modelagem

Referência para a regra EPV (events per variable):
PEDUZZI, P.; CONCATO, J.; KEMPER, E.; HOLFORD, T. R.; FEINSTEIN, A. R.
A simulation study of the number of events per variable in logistic
regression analysis. Journal of Clinical Epidemiology, v. 49, n. 12,
p. 1373-1379, 1996. DOI: 10.1016/S0895-4356(96)00236-3.

Ressalva metodológica: a rigidez da regra "10-20 eventos por variável"
é questionada por literatura mais recente:
VAN SMEDEN, M. et al. No rationale for 1 variable per 10 events criterion
for binary logistic regression analysis. BMC Medical Research Methodology,
v. 16, 2016.
A folga observada abaixo (ver print de "folga sobre o mínimo exigido")
mitiga essa limitação ao ultrapassar o mínimo com margem considerável.

Nota: a etapa de amostragem em si não faz parte do procedimento clássico
de Fávero (Manual de Análise de Dados), que trabalha com datasets já
pequenos. É uma adaptação necessária por este projeto partir de uma base
administrativa de escala nacional (CNPJ/RFB).
"""

import pandas as pd

CAMINHO_ENTRADA = "data/cohort_varejo_completo.parquet"
CAMINHO_SAIDA = "data/amostra_varejo_60k.parquet"

TAMANHO_AMOSTRA = 60_000
SEED = 42

PARAMETROS_ESTIMADOS = 20
EPV_MINIMO = 20  # critério mais rigoroso da faixa 10-20 sugerida por Peduzzi et al. (1996)

df = pd.read_parquet(CAMINHO_ENTRADA)

# Remove Nula (01) e Suspensa (03) — fora do escopo da variável resposta
df = df[df["situacao_cadastral"].isin(["02", "04", "08"])].copy()

# Variável resposta binária: 1 = risco (baixada/inapta), 0 = ativa
df["risco"] = df["situacao_cadastral"].isin(["04", "08"]).astype(int)

proporcao_ativa = (df["risco"] == 0).mean()
print(f"Proporção da classe minoritária (ativa) na população: {proporcao_ativa:.2%}")

casos_minoritarios_necessarios = PARAMETROS_ESTIMADOS * EPV_MINIMO
print(f"Casos mínimos necessários da classe minoritária (regra EPV): {casos_minoritarios_necessarios}")

# Amostragem estratificada proporcional: usa .sample() do GroupBy (não .apply()),
# que preserva todas as colunas — inclusive a própria coluna de agrupamento.
# A mesma fração aplicada a cada grupo mantém a proporção original da população.
fracao = TAMANHO_AMOSTRA / len(df)
amostra = df.groupby("risco").sample(frac=fracao, random_state=SEED)

casos_minoritarios_amostra = (amostra["risco"] == 0).sum()
folga = casos_minoritarios_amostra / casos_minoritarios_necessarios

print(f"\nTamanho da amostra: {len(amostra)}")
print(f"Casos da classe minoritária (ativa) na amostra: {casos_minoritarios_amostra}")
print(f"Folga sobre o mínimo exigido (regra EPV): {folga:.1f}x")

print("\nProporção na amostra:")
print(amostra["risco"].value_counts(normalize=True))

amostra.to_parquet(CAMINHO_SAIDA, index=False)
print(f"\nSalvo em {CAMINHO_SAIDA}")