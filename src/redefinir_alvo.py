"""
Define o alvo do modelo de risco de encerramento/irregularidade para
varejistas (CNAE divisão 47), a partir de `situacao_cadastral` e
`motivo_situacao_cadastral` (Receita Federal, Dados Abertos de CNPJ,
extração 08/08/2026), cruzados com a tabela oficial de Motivos
(data/raw/Motivos.zip).

Por que isso é necessário
-------------------------
A coluna `risco` original (0 = ATIVA, 1 = INAPTA ou BAIXADA agrupados)
misturava dois fenômenos diferentes:
  - encerramento voluntário/real do negócio -- a empresa deixou de operar,
    o que é o evento de "fracasso" relevante para uma decisão de crédito
    (deixou de gerar receita e capacidade de pagamento);
  - irregularidade declaratória -- INAPTA significa que a empresa deixou de
    entregar declarações à Receita, não necessariamente que fechou. Tratar
    isso como "fracasso" contaminaria o modelo com um fenômeno distinto
    (compliance fiscal, não saúde do negócio).

Além disso, `situacao_cadastral == '08'` (BAIXADA) não é uma categoria
única: o código `motivo_situacao_cadastral` registra motivos tão diferentes
quanto liquidação voluntária, falência, incorporação societária ou óbito do
titular MEI. Cada um desses motivos foi conferido um a um contra a tabela
oficial de Motivos da Receita antes de decidir o bucket abaixo -- nenhuma
suposição foi feita sem checar a descrição oficial do código.

Os 60.000 estabelecimentos da amostra são separados em 4 buckets
mutuamente exclusivos:

1) ALVO PRINCIPAL, alvo_baixada = 0 (sobreviveu):
   `situacao_cadastral == '02'` (ATIVA).

2) ALVO PRINCIPAL, alvo_baixada = 1 (fracassou):
   `situacao_cadastral == '08'` (BAIXADA) com motivo em:
     01 - EXTINCAO POR ENCERRAMENTO LIQUIDACAO VOLUNTARIA
     05 - ENCERRAMENTO DA FALENCIA
     06 - ENCERRAMENTO DA LIQUIDACAO
     10 - EXTINCAO PELO ENCERRAMENTO DA LIQUIDACAO JUDICIAL
     54 - EXTINCAO - TRATAMENTO DIFERENCIADO ME/EPP (LC 123/2006)
   Todos são encerramentos reais e definitivos do negócio: é exatamente o
   evento que um modelo de risco de crédito precisa antecipar.

3) BUCKET DE ROBUSTEZ (INAPTA -- análise separada de "o que prediz
   inaptidão", fora do alvo principal):
   `situacao_cadastral == '04'` (INAPTA)
   OU `situacao_cadastral == '08'` com motivo em:
     66 - INAPTIDAO
     73 - COOMISSAO CONTUMAZ (omissão contumaz de declarações)
   Mesma família fenomenológica da INAPTA -- irregularidade declaratória,
   não necessariamente o fim do negócio --, só que registrada sob BAIXADA
   em vez de INAPTA. Fica isolado para uma investigação futura de "o que
   prediz inaptidão", sem contaminar o alvo principal de crédito.

4) EXCLUÍDOS COMPLETAMENTE (ruído para esta pergunta -- não são fracasso de
   negócio nem irregularidade declaratória):
   `situacao_cadastral == '08'` com motivo em:
     02 - INCORPORACAO            (virou outra empresa, não fechou)
     15 - INEXISTENCIA DE FATO    (nunca existiu de fato, não houve encerramento)
     67 - REGISTRO CANCELADO      (problema registral, não é evento de negócio)
     75 - OBITO DO MEI - TITULAR FALECIDO (evento pessoal, não é risco de negócio)
   Saem do dataset de modelagem inteiramente -- não entram nem no alvo
   principal, nem no bucket de robustez.

Saídas
------
- data/dataset_modelagem_baixada.parquet: buckets 1 + 2 (alvo principal),
  com a coluna `alvo_baixada` preenchida.
- data/dataset_excluidos_robustez.parquet: bucket 3 (robustez/INAPTA).
- data/dataset_excluidos_ruido.parquet: bucket 4 (ruído), com a coluna
  `motivo_exclusao` explicando por que cada caso saiu.
"""
import pandas as pd

CAMINHO_ENTRADA = "data/dataset_modelagem.parquet"
CAMINHO_SAIDA_ALVO = "data/dataset_modelagem_baixada.parquet"
CAMINHO_SAIDA_ROBUSTEZ = "data/dataset_excluidos_robustez.parquet"
CAMINHO_SAIDA_RUIDO = "data/dataset_excluidos_ruido.parquet"

MOTIVOS_FRACASSO = ["01", "05", "06", "10", "54"]
MOTIVOS_ROBUSTEZ = ["66", "73"]
MOTIVOS_RUIDO = {
    "02": "incorporação — virou outra empresa, não é encerramento",
    "15": "inexistência de fato — nunca existiu de fato, não houve encerramento",
    "67": "registro cancelado — problema registral, não é evento de negócio",
    "75": "óbito do titular MEI — evento pessoal, não relacionado a risco de negócio",
}

df = pd.read_parquet(CAMINHO_ENTRADA)
print(f"Dataset original: {df.shape}")

print("\nDistribuição original (situacao_cadastral):")
print(df["situacao_cadastral"].value_counts())

ativa = df["situacao_cadastral"] == "02"
baixada_fracasso = (df["situacao_cadastral"] == "08") & (
    df["motivo_situacao_cadastral"].isin(MOTIVOS_FRACASSO)
)
robustez = (df["situacao_cadastral"] == "04") | (
    (df["situacao_cadastral"] == "08") & (df["motivo_situacao_cadastral"].isin(MOTIVOS_ROBUSTEZ))
)
ruido = (df["situacao_cadastral"] == "08") & (
    df["motivo_situacao_cadastral"].isin(MOTIVOS_RUIDO)
)

# Os 4 buckets devem ser mutuamente exclusivos e cobrir os 60.000 originais.
soma_buckets = ativa.sum() + baixada_fracasso.sum() + robustez.sum() + ruido.sum()
assert soma_buckets == len(df), (
    f"Buckets não cobrem o dataset inteiro: soma={soma_buckets}, total={len(df)}"
)
sobreposicao = (
    (ativa & baixada_fracasso)
    | (ativa & robustez)
    | (ativa & ruido)
    | (baixada_fracasso & robustez)
    | (baixada_fracasso & ruido)
    | (robustez & ruido)
)
assert not sobreposicao.any(), "Buckets se sobrepõem -- revise os critérios."
print(f"\nConferido: os 4 buckets somam {soma_buckets:,} e são mutuamente exclusivos.")

# --- Bucket 1+2: alvo principal ---
dataset_alvo = df[ativa | baixada_fracasso].copy()
dataset_alvo["alvo_baixada"] = baixada_fracasso[ativa | baixada_fracasso].astype(int)

print(f"\nAlvo principal: {dataset_alvo.shape}")
print(dataset_alvo["alvo_baixada"].value_counts())
print(dataset_alvo["alvo_baixada"].value_counts(normalize=True))

dataset_alvo.to_parquet(CAMINHO_SAIDA_ALVO, index=False)
print(f"Salvo em {CAMINHO_SAIDA_ALVO}")

# --- Bucket 3: robustez (INAPTA) ---
dataset_robustez = df[robustez].copy()
print(f"\nBucket de robustez (INAPTA): {dataset_robustez.shape}")
dataset_robustez.to_parquet(CAMINHO_SAIDA_ROBUSTEZ, index=False)
print(f"Salvo em {CAMINHO_SAIDA_ROBUSTEZ}")

# --- Bucket 4: ruído ---
dataset_ruido = df[ruido].copy()
dataset_ruido["motivo_exclusao"] = dataset_ruido["motivo_situacao_cadastral"].map(MOTIVOS_RUIDO)
print(f"\nExcluídos como ruído: {dataset_ruido.shape}")
print(dataset_ruido["motivo_exclusao"].value_counts())
dataset_ruido.to_parquet(CAMINHO_SAIDA_RUIDO, index=False)
print(f"Salvo em {CAMINHO_SAIDA_RUIDO}")

print("\nResumo final dos 4 buckets:")
print(f"  ATIVA (alvo=0):              {ativa.sum():>7,}")
print(f"  BAIXADA fracasso (alvo=1):   {baixada_fracasso.sum():>7,}")
print(f"  Robustez (INAPTA):           {robustez.sum():>7,}")
print(f"  Ruído (excluído):            {ruido.sum():>7,}")
print(f"  Total:                       {soma_buckets:>7,}")
