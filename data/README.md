# data/ — pipeline de dados do modelo de risco de baixada (varejo)

Guarda as saídas intermediárias e finais do pipeline, da coorte consolidada de varejo até
os artefatos do modelo treinado. A origem bruta (Receita Federal) e os primeiros
cruzamentos estão documentados em [`raw/README.md`](raw/README.md).

Ordem de geração:

`raw/` (scripts) → `cohort_varejo_completo.parquet` → `amostra_varejo_60k.parquet` →
`dataset_modelagem.parquet` → `dataset_modelagem_baixada.parquet` (+ `dataset_excluidos_robustez.parquet`
e `dataset_excluidos_ruido.parquet`) → `dataset_modelagem_final.parquet` (notebooks 02-04) →
`df_modelo_treino.parquet` + `modelo_final_logit.pickle` + `probs_treino.parquet` (notebook 05)

## Arquivos

### `cohort_varejo_completo.parquet`
- **O que é:** coorte completa de comércio varejista (CNAE `47*`) aberto entre 2017 e
  2019, com todas as situações cadastrais (ativa, baixada, inapta, suspensa, nula).
- **Origem:** produzido por [`../src/situacao_cadastral.py`](../src/situacao_cadastral.py),
  que consolida os 10 arquivos parciais de
  [`raw/estabelecimento/cohort/`](raw/README.md).
- **Tamanho:** 1.911.883 linhas, 30 colunas.
- **Consumido por:** [`../src/preparacao_amostra.py`](../src/preparacao_amostra.py),
  [`../src/cruzamento_empresas.py`](../src/cruzamento_empresas.py),
  [`../src/cruzamento_simples.py`](../src/cruzamento_simples.py).

### `amostra_varejo_60k.parquet`
- **O que é:** amostra estratificada de 60.000 estabelecimentos da coorte completa, já
  com a variável binária `risco` (0 = ativa, 1 = baixada/inapta) e as situações fora de
  escopo (nula, suspensa) removidas. Tamanho definido pela regra EPV (events per
  variable) — justificativa completa no docstring do script.
- **Origem:** produzido por
  [`../src/preparacao_amostra.py`](../src/preparacao_amostra.py), a partir de
  `cohort_varejo_completo.parquet`.
- **Tamanho:** 60.000 linhas, 31 colunas.
- **Consumido por:** [`../src/dataset_modelagem.py`](../src/dataset_modelagem.py).

### `dataset_modelagem.parquet`
- **O que é:** a amostra de 60 mil casos já com as variáveis explicativas de Empresas
  (natureza jurídica, capital social, porte) e Simples/MEI juntadas pelo `cnpj_basico`.
- **Origem:** produzido por [`../src/dataset_modelagem.py`](../src/dataset_modelagem.py),
  a partir de `amostra_varejo_60k.parquet`, `raw/empresa/cohort_empresas.parquet` e
  `raw/simples/cohort_simples.parquet`.
- **Tamanho:** 60.000 linhas, 43 colunas.
- **Consumido por:** [`../src/redefinir_alvo.py`](../src/redefinir_alvo.py).

### `dataset_modelagem_baixada.parquet`
- **O que é:** o alvo principal do modelo — só os casos ATIVA (`alvo_baixada=0`) e
  BAIXADA por encerramento real do negócio (`alvo_baixada=1`), sem os casos INAPTA e sem
  os motivos de baixada considerados ruído. Lista completa de motivos em cada bucket no
  docstring de `redefinir_alvo.py`.
- **Origem:** produzido por [`../src/redefinir_alvo.py`](../src/redefinir_alvo.py)
  (bucket 1+2), a partir de `dataset_modelagem.parquet`.
- **Tamanho:** 46.454 linhas, 44 colunas.
- **Consumido por:** [`../notebooks/01_exploracao_preditores.ipynb`](../notebooks/01_exploracao_preditores.ipynb)
  e [`../notebooks/02_exclusao_artefato_eireli.ipynb`](../notebooks/02_exclusao_artefato_eireli.ipynb).

### `dataset_excluidos_robustez.parquet`
- **O que é:** casos INAPTA (irregularidade declaratória, não necessariamente
  encerramento do negócio) — deixados fora do alvo principal para não misturar
  compliance fiscal com risco de negócio, mas guardados para uma checagem de robustez.
- **Origem:** produzido por [`../src/redefinir_alvo.py`](../src/redefinir_alvo.py)
  (bucket 3), a partir de `dataset_modelagem.parquet`.
- **Tamanho:** 12.822 linhas, 43 colunas.
- **Consumido por:** [`../notebooks/05_estimacao_modelo.ipynb`](../notebooks/05_estimacao_modelo.ipynb)
  (prévia de shape/colunas) e [`../notebooks/06_analise_robustez_inapta.ipynb`](../notebooks/06_analise_robustez_inapta.ipynb)
  (análise completa).

### `dataset_excluidos_ruido.parquet`
- **O que é:** casos de baixada por motivos que não são nem fracasso de negócio nem
  irregularidade declaratória (incorporação, inexistência de fato, registro cancelado,
  óbito do titular MEI) — ruído para esta pergunta de pesquisa. Tem a coluna extra
  `motivo_exclusao` explicando cada caso.
- **Origem:** produzido por [`../src/redefinir_alvo.py`](../src/redefinir_alvo.py)
  (bucket 4), a partir de `dataset_modelagem.parquet`.
- **Tamanho:** 724 linhas, 44 colunas.
- **Consumido por:** nenhum notebook ou script no momento — mantido como registro de
  auditoria de quem foi excluído e por quê.

### `dataset_modelagem_final.parquet`
- **O que é:** o dataset de modelagem definitivo — mesmo arquivo, reescrito em sequência
  por três notebooks, cada um adicionando ou removendo algo específico:
  1. [`02_exclusao_artefato_eireli.ipynb`](../notebooks/02_exclusao_artefato_eireli.ipynb):
     lê `dataset_modelagem_baixada.parquet`, remove as 578 linhas de EIRELI (artefato de
     reclassificação legal) → 44 colunas.
  2. [`03_agrupamento_natureza_juridica.ipynb`](../notebooks/03_agrupamento_natureza_juridica.ipynb):
     adiciona `natureza_juridica_agrupada` → 45 colunas.
  3. [`04_agrupamento_uf_cnae.ipynb`](../notebooks/04_agrupamento_uf_cnae.ipynb): adiciona
     `regiao` e `grupo_cnae`, depois remove as 2 linhas com `uf='EX'` (empresas no
     exterior, sem região brasileira válida) → shape final.
- **Tamanho final:** 45.874 linhas, 47 colunas.
- **Consumido por:** os próprios notebooks 03 e 04 (cada um lê o checkpoint deixado pelo
  anterior) e [`../notebooks/05_estimacao_modelo.ipynb`](../notebooks/05_estimacao_modelo.ipynb).

### `perfil_colunas_dataset_modelagem.csv`
- **O que é:** perfil coluna a coluna (tipo, % de nulos, cardinalidade, valores mais
  comuns) de `dataset_modelagem_baixada.parquet`, gerado para orientar a escolha de
  preditores na exploração inicial.
- **Origem:** produzido por [`../notebooks/01_exploracao_preditores.ipynb`](../notebooks/01_exploracao_preditores.ipynb).
- **Tamanho:** 44 linhas (uma por coluna do dataset perfilado), 5 colunas.
- **Consumido por:** nenhum notebook ou script — artefato de consulta humana.

### `df_modelo_treino.parquet`
- **O que é:** a base de treino no formato usado para ajustar o modelo — os 7 preditores
  finais (já transformados: `capital_social_log` em vez de `capital_social`) mais a
  coluna `alvo`, antes da codificação em dummies.
- **Origem:** produzido por [`../notebooks/05_estimacao_modelo.ipynb`](../notebooks/05_estimacao_modelo.ipynb),
  a partir de `dataset_modelagem_final.parquet`.
- **Tamanho:** 45.874 linhas, 8 colunas.
- **Consumido por:** [`../notebooks/06_analise_robustez_inapta.ipynb`](../notebooks/06_analise_robustez_inapta.ipynb).

### `modelo_final_logit.pickle`
- **O que é:** o modelo de regressão logística final (statsmodels), já após a seleção de
  variáveis por stepwise manual — 18 preditores (dummies incluídos).
- **Origem:** produzido por [`../notebooks/05_estimacao_modelo.ipynb`](../notebooks/05_estimacao_modelo.ipynb).
- **Consumido por:** [`../notebooks/06_analise_robustez_inapta.ipynb`](../notebooks/06_analise_robustez_inapta.ipynb),
  que carrega o modelo para pontuar os casos INAPTA sem re-treinar.

### `probs_treino.parquet`
- **O que é:** o alvo real (`alvo`) e a probabilidade de baixada prevista pelo modelo
  final (`prob_baixada_prevista`) para cada caso do treino — usado para comparar contra
  a probabilidade prevista dos casos INAPTA.
- **Origem:** produzido por [`../notebooks/05_estimacao_modelo.ipynb`](../notebooks/05_estimacao_modelo.ipynb).
- **Tamanho:** 45.874 linhas, 2 colunas.
- **Consumido por:** [`../notebooks/06_analise_robustez_inapta.ipynb`](../notebooks/06_analise_robustez_inapta.ipynb)
  (boxplot comparando ATIVA/INAPTA/BAIXADA).
