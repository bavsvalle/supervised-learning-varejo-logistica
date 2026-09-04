import pandas as pd
import zipfile
import os

# Primeira etapa do pipeline: varre os 10 arquivos brutos de Estabelecimentos da Receita Federal
# (divididos pela própria RFB) e já filtra, por arquivo, só o que interessa para este projeto --
# processar tudo (dezenas de milhões de linhas) sem filtro antes seria caro demais em disco/memória.
pasta_raw = "data/raw"

# layout oficial do arquivo ESTABELECIMENTOS da RFB (sem cabeçalho, ; como separador, latin-1) --
# só as colunas realmente usadas pelo projeto foram nomeadas aqui
colunas_estabelecimentos = [
    "cnpj_basico", "cnpj_ordem", "cnpj_dv", "identificador_matriz_filial",
    "nome_fantasia", "situacao_cadastral", "data_situacao_cadastral",
    "motivo_situacao_cadastral", "nome_cidade_exterior", "pais",
    "data_inicio_atividade", "cnae_fiscal_principal", "cnae_fiscal_secundaria",
    "tipo_logradouro", "logradouro", "numero", "complemento", "bairro",
    "cep", "uf", "municipio", "ddd1", "telefone1", "ddd2", "telefone2",
    "ddd_fax", "fax", "correio_eletronico", "situacao_especial",
    "data_situacao_especial"
]

def processar_arquivo(caminho_csv, indice):
    lista_cohorts = []
    # le em chunks de 200k linhas -- os CSVs brutos da RFB tem dezenas de milhoes de linhas
    # cada, carregar tudo de uma vez estouraria memoria
    for chunk in pd.read_csv(
        caminho_csv, sep=";", encoding="latin1", header=None,
        names=colunas_estabelecimentos, dtype=str, chunksize=200_000
    ):
        # CNAE fiscal principal comecando em "47" = divisao "Comercio varejista" (escopo do projeto)
        varejo = chunk[chunk["cnae_fiscal_principal"].str.startswith("47", na=False)]
        # restringe a coorte de empresas abertas entre 2017 e 2019 -- da tempo suficiente
        # (ate a extracao em 2026) para observar se a empresa sobreviveu ou fracassou
        cohort = varejo[
            (varejo["data_inicio_atividade"] >= "20170101") &
            (varejo["data_inicio_atividade"] <= "20191231")
        ]
        if len(cohort) > 0:
            lista_cohorts.append(cohort)

    if lista_cohorts:
        df_cohort = pd.concat(lista_cohorts, ignore_index=True)
        caminho_saida = f"{pasta_raw}/cohort_estabelecimentos{indice}.parquet"
        df_cohort.to_parquet(caminho_saida, index=False)
        print(f"Estabelecimentos{indice}: {len(df_cohort)} linhas salvas em {caminho_saida}")
    else:
        print(f"Estabelecimentos{indice}: nenhuma linha no cohort")

# a RFB distribui Estabelecimentos em 10 arquivos zip (0 a 9); processa um de cada vez
for indice in range(10):
    caminho_parquet = f"{pasta_raw}/cohort_estabelecimentos{indice}.parquet"
    if os.path.exists(caminho_parquet):
        # idempotente: se o parquet ja existe, nao reprocessa o zip de novo (evita retrabalho caro)
        print(f"Estabelecimentos{indice}: já processado anteriormente, pulando.")
        continue

    nome_zip = f"{pasta_raw}/Estabelecimentos{indice}.zip"
    if not os.path.exists(nome_zip):
        print(f"Estabelecimentos{indice}.zip não encontrado em {pasta_raw} — baixe do link da RFB antes de rodar de novo.")
        continue

    with zipfile.ZipFile(nome_zip) as z:
        nome_interno = z.namelist()[0]
        z.extract(nome_interno, path=pasta_raw)
        caminho_extraido = os.path.join(pasta_raw, nome_interno)
        processar_arquivo(caminho_extraido, indice)
        # remove o CSV bruto extraido (varios GB) apos processar -- so o parquet filtrado fica em disco
        os.remove(caminho_extraido)