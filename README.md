# Regressão Logística - Risco de Encerramento no Varejo

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-black?logo=pandas)
![Statsmodels](https://img.shields.io/badge/Statsmodels-red)
![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?logo=duckdb&logoColor=black)

## 🎯 Sobre o projeto

**O problema.** Fintechs que emprestam para pequenos varejistas muitas vezes não têm balanço para analisar. Quem aprova o crédito precisa de algum sinal de risco que exista antes de a empresa ter histórico próprio.

**O que o projeto entrega.** Um modelo de regressão logística que estima, usando só dados cadastrais públicos da Receita Federal, a probabilidade de um estabelecimento varejista ser baixado por encerramento (liquidação, falência ou extinção). A ideia é usar essa probabilidade como mais um sinal de risco, ao lado das informações que a fintech já tem, e não como substituto da análise de crédito.

**Quem usaria.** Uma área de crédito ou de risco de uma fintech que atende pequenos varejistas. O projeto usa dados públicos e não tem cliente real, então esse uso é um cenário, não um caso aplicado.

**O resultado.** Em uma amostra de teste separada, o modelo tem AUC de 0,65 (validação cruzada: 0,649, desvio de 0,005), ou seja, discrimina de forma moderada. Com corte de 0,62, ele identifica cerca de 66% dos estabelecimentos que foram baixados, ao custo de apontar como risco cerca de 44% dos que seguem ativos. Ser filial (em vez de matriz) e ter menor capital social estão associados a maior risco de encerramento, mas o efeito de filial é menos firme do que parece (veja [Limitações](#️-limitações)).

**Ressalvas importantes:** baixa não é o mesmo que inadimplência, e o modelo usa só dado cadastral, sem histórico financeiro. Por isso ele serve como sinal complementar, não como decisor.

---

## 📖 Etapas do Projeto

### 1. Coleta dos Dados

- **Fonte:** [Dados Abertos de CNPJ](https://dados.gov.br/dados/conjuntos-dados/cadastro-nacional-da-pessoa-juridica---cnpj), Receita Federal (extração de referência: 08/08/2026)
- **Arquivos:** Estabelecimentos, Empresas, Simples/MEI, Motivos e Naturezas (tabelas oficiais de referência código → descrição)
- **Coorte:** estabelecimentos com CNAE principal na divisão 47 (Comércio Varejista), abertos entre 2017 e 2019, num total de 1.911.883
- **Amostra para modelagem:** 60.000 estabelecimentos, com estratificação proporcional por risco (preserva a taxa de baixada, não equilibra as classes). Fixei o tamanho em 60.000 e conferi contra a regra de EPV (Peduzzi et al., 1996).
- **Dataset final de modelagem:** 45.874 estabelecimentos (59% baixados), depois de separar os grupos de robustez e ruído e de remover os artefatos descritos abaixo.

### 2. Metodologia

- Defini o alvo em 4 buckets (ATIVA, BAIXADA, INAPTA e RUÍDO), cruzando `situacao_cadastral` e `motivo_situacao_cadastral` com a tabela oficial de Motivos, em vez de assumir a classificação.
- Encontrei dois casos de vazamento de dado e tirei essas variáveis do modelo:
  - **EIRELI** (natureza jurídica 2305): 578 casos, todos baixados e nenhum ativo. A explicação que adotei é a extinção da EIRELI pela Lei 14.195/2021, que converteu as sobreviventes em outra forma societária. Não testei diretamente essa migração.
  - **`optante_simples` e `optante_mei`:** a opção é cancelada junto com a baixa do CNPJ (a data de exclusão do Simples coincide com a da baixa, mediana de 0 dia).
- Agrupei categorias raras (`natureza_juridica_agrupada`, `regiao`, `grupo_cnae` com 8 grupos de 3 dígitos) e fiz a dummização usando a categoria mais frequente como referência. Detalhes nos notebooks 03 e 04.
- Estimei o modelo completo com 20 termos e depois fiz o **Stepwise manual** (backward elimination pelo teste de Wald, só com `statsmodels`). Tive que implementar na mão porque o `statstests.process.stepwise` entrou em loop infinito no meu ambiente.
- No diagnóstico usei matriz de confusão, sensitividade e especificidade por cutoff, curva ROC, AUC e GINI.
- Fiz uma **análise de robustez** no bucket INAPTA, que é uma extensão minha e não foi vista no material do curso.
- Fiz uma **validação fora da amostra** (notebook 07): reestimei o modelo (inclusive o Stepwise) numa amostra de treino e avaliei numa amostra de teste separada, além de validação cruzada em 5 partes e checagem de calibração.

### 3. Resultado do Modelo Final

| Modelo | Descrição | Pseudo R² | AUC* | GINI* |
|---|---|---|---|---|
| `modelo_completo` | 20 termos, antes do Stepwise | 0,0625 | não calculado | não calculado |
| `modelo_step_logit` | **Modelo final**: Stepwise manual, 18 termos (7 variáveis) | 0,0624 | 0,6521 | 0,3042 |
| `modelo_step_robustez` | Refit com o bucket INAPTA tratado como baixada (19 termos) | 0,0535 | 0,6415 | 0,2829 |

\* Calculados na própria amostra de estimação. A validação fora da amostra está logo abaixo.

O modelo final ficou com 18 dos 20 termos, todos significativos a 5%. `regiao_Norte` e `regiao_Sul` saíram no Stepwise e o Pseudo R² quase não mudou (0,0625 para 0,0624). O Pseudo R² baixo é esperado em regressão logística com dado cadastral, e a discriminação moderada faz sentido para um modelo sem histórico financeiro.

**Validação fora da amostra (notebook 07):**

| Medida | Valor |
|---|---|
| AUC no teste | 0,6545 |
| GINI no teste | 0,309 |
| KS no teste | 0,234 |
| AUC em validação cruzada (5 partes) | 0,649 (desvio de 0,005) |
| Corte de 0,62: sensitividade / especificidade | 0,66 / 0,56 |
| Corte de 0,5: proporção classificada como baixada / especificidade | 82% / 0,30 |

O AUC no teste (0,6545) ficou praticamente igual ao da amostra de estimação (0,6521), o que indica que o modelo não está inflado por sobreajuste.

![Curva ROC](outputs/curva_roc.png)

**Principais achados:**

- **Filial.** Ser filial tem o maior efeito bruto: coeficiente de +1,17 em relação à matriz, o que dá uma chance de encerramento cerca de 3,2 vezes maior, com as demais variáveis constantes. Esse efeito, porém, depende da natureza jurídica (veja Limitações).
- **Capital social.** O coeficiente de `capital_social_log` é negativo: quanto maior o capital social, menor o risco.
- **Peso relativo.** Em escala padronizada, a natureza jurídica 2062 (sociedade empresária limitada) pesa mais (0,44), seguida de filial (0,27) e de capital social (−0,19).
- **Cutoff.** No padrão de 0,5 o modelo classifica baixada demais (especificidade de apenas 0,30). O ponto de equilíbrio entre sensitividade e especificidade fica entre 0,60 e 0,65, e usei 0,62 na validação.
- **Calibração.** As probabilidades são bem calibradas no centro, mas o último decil superestima o risco (0,755 previsto contra 0,709 observado).

![Sensitividade e especificidade por cutoff](outputs/sensitividade_especificidade_cutoff.png)

---

## 🔍 Análise de Robustez (bucket INAPTA)

O bucket INAPTA (12.822 estabelecimentos) mistura empresas em situação cadastral inapta com baixadas por motivos ambíguos (66 e 73). Ficou fora do treino porque não é claramente ativa nem baixada. Para ver se isso mudava alguma coisa, apliquei o modelo final nessas empresas (`.predict()`) e comparei com os grupos reais:

![Boxplot de robustez](outputs/robustez_boxplot_inapta.png)

| Grupo | Probabilidade prevista (mediana) |
|---|---|
| ATIVA (real) | 0,605 |
| INAPTA (excluída) | 0,638 |
| BAIXADA (real) | 0,646 |

A mediana da INAPTA fica mais perto da BAIXADA do que da ATIVA, mas as três são próximas. As medianas escondem parte da história: a distribuição da ATIVA é bimodal (quartis em 0,385 e 0,658), o que enfraquece a comparação só por medianas.

Depois refiz o Stepwise do zero tratando a INAPTA como alvo = 1 (linha `modelo_step_robustez` da tabela). Comparando os summaries dos dois modelos, nenhum coeficiente de preditor trocou de sinal (só o intercepto muda, e `regiao_Sul` volta como 19º termo no modelo de robustez), e o AUC caiu de 0,6521 para 0,6415 (1,6% em termos relativos). Ou seja, excluir a INAPTA não mudou as conclusões do modelo.

---

## ⚠️ Limitações

- **Só dado cadastral público.** Sem histórico financeiro ou comportamental, o poder de discriminação tem um teto baixo. Modelos com dados de bureau de crédito costumam discriminar melhor.
- **Baixa não é inadimplência.** O alvo mede encerramento cadastral, e entre os motivos estão liquidação voluntária e extinção de ME/EPP, que nem sempre significam que o negócio fracassou.
- **O efeito de filial é frágil.** O coeficiente cai de 1,17 para 0,30 quando as linhas da natureza jurídica 2062 saem da base. Só 5% dos estabelecimentos (2.325) são filiais. Dentro da natureza jurídica 2135 (empresário individual, a categoria de referência) a diferença entre filial e matriz é de cerca de 2 pontos percentuais, com apenas 275 filiais. Tratar "ser filial" como o principal fator de risco exige cautela.
- **Dependência entre observações.** Alguns CNPJs aparecem em mais de uma linha (142 CNPJs básicos repetidos, somando 406 linhas, quase todas filiais). O modelo usa erros padrão comuns, sem agrupar por CNPJ, então os p-valores podem ser otimistas.
- **O Stepwise não é estável.** Nas partes da validação cruzada o modelo manteve entre 14 e 18 termos, e no treino o Stepwise removeu 3 termos contra 2 na base completa. A lista exata de variáveis não deve ser lida como definitiva.
- **Previsões concentradas.** A maior parte das probabilidades fica entre 0,6 e 0,7, e no corte de 0,5 o modelo classifica 82% como baixada. O ganho de previsão sobre simplesmente "chutar a média" é pequeno (Brier skill de 0,088).
- **Categorias muito pequenas.** "Outras" tem 20 linhas, e as naturezas 2054 e 2046 têm cerca de 100 cada, com intervalos de confiança largos.
- **Capital social com valores extremos.** Há 44 linhas com capital menor ou igual a zero e valores máximos de ordem de 10¹⁰. O `log1p` suaviza, mas os extremos continuam na base.
- **Janela de exposição.** `idade_empresa_anos` é calculada contra uma data fixa (08/08/2026) para toda a coorte, e a faixa de idade é estreita (6,6 a 9,6 anos). O coeficiente positivo pode refletir em parte o tempo de exposição ao risco, e não uma fragilidade maior de empresas mais antigas.
- **Associação, não causalidade.** Os coeficientes descrevem padrões da coorte 2017-2019 observada em 2026.

**Próximo passo:** repetir a validação agrupando por CNPJ e testar erros padrão robustos a esse agrupamento.

---

## ⚙️ Como Rodar

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_exploracao_preditores.ipynb
```

Os notebooks devem ser executados em ordem (01 → 07), porque cada um usa os arquivos salvos pelo anterior. Os notebooks 02, 03 e 04 leem os dados a partir da raiz do repositório, e os demais a partir da pasta `notebooks/`, então confira o diretório de trabalho de cada um.

**Ambientes:** os notebooks 01 a 06 foram executados em Python 3.11 com pandas 3.0.5, numpy 2.4.6, statsmodels 0.14.6 e scikit-learn 1.9.0. O notebook 07 foi executado em um ambiente separado, com as versões do `requirements.txt` (pandas 2.3.3, numpy 2.2.6, statsmodels 0.15.0 e scikit-learn 1.7.2). Ao reproduzir, pequenas diferenças numéricas são possíveis.

**Dados:** os parquets processados em `data/` estão versionados, então os notebooks rodam sem baixar nada. A pasta `data/raw/` (extrações brutas) e o `cohort_varejo_completo.parquet` (151 MB) não estão no repositório. Para regenerá-los, baixe os arquivos da Receita Federal (link na seção de coleta) e rode os scripts de `src/` nesta ordem: `situacao_cadastral.py` → `preparacao_amostra.py` → `dataset_modelagem.py` → `redefinir_alvo.py`. Detalhes em `data/README.md`.

<details>
<summary><b>📁 Estrutura do repositório</b></summary>

    supervised-learning-varejo-logistica/
    ├── data/
    │   ├── README.md
    │   ├── amostra_varejo_60k.parquet
    │   ├── dataset_modelagem.parquet
    │   ├── dataset_modelagem_baixada.parquet
    │   ├── dataset_modelagem_final.parquet
    │   ├── dataset_excluidos_robustez.parquet
    │   ├── dataset_excluidos_ruido.parquet
    │   ├── df_modelo_treino.parquet
    │   ├── probs_treino.parquet
    │   ├── perfil_colunas_dataset_modelagem.csv
    │   └── modelo_final_logit.pickle
    ├── notebooks/
    │   ├── 01_exploracao_preditores.ipynb
    │   ├── 02_exclusao_artefato_eireli.ipynb
    │   ├── 03_agrupamento_natureza_juridica.ipynb
    │   ├── 04_agrupamento_uf_cnae.ipynb
    │   ├── 05_estimacao_modelo.ipynb
    │   ├── 06_analise_robustez_inapta.ipynb
    │   └── 07_validacao.ipynb
    ├── src/
    │   ├── pipeline_extracao.py
    │   ├── cruzamento_empresas.py
    │   ├── cruzamento_simples.py
    │   ├── situacao_cadastral.py
    │   ├── preparacao_amostra.py
    │   ├── dataset_modelagem.py
    │   └── redefinir_alvo.py
    ├── outputs/
    │   ├── curva_roc.png
    │   ├── sensitividade_especificidade_cutoff.png
    │   └── robustez_boxplot_inapta.png
    ├── requirements.txt
    ├── .gitignore
    └── README.md

</details>