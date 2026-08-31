# %%
import pandas as pd
from bacendata import sgs

def coletar_fluxo_capitais(start_date='2000-01-01', end_date='2026-03-31'):
    """
        Coleta os dados de fluxo de capitais do Banco Central do Brasil (BCB) usando a biblioteca bacendata.
        Parâmetros:
        start_date (str): Data de início no formato 'YYYY-MM-DD'.
        end_date (str): Data de término no formato 'YYYY-MM-DD'.
        
        Retorna:
        pd.DataFrame: DataFrame contendo os dados de fluxo de capitais.
    """

    series = {
        'meta_taxa_selic': 432,
        'taxa_selic': 11,
        'taxa_cdi': 12,
        'dolar_cambio_livre': 1, #venda (quanto esta vendendo para comprar real)
        'euro_cambio_livre': 21619,
        'iene_cambio_livre': 21621
        
    }
    dados_fluxo = {}
    for nome, codigo in series.items():
        try:
            serie = sgs.get(codigo, start=start_date, end=end_date)
            serie = serie.rename(columns={codigo: nome})
            
            print(f"{nome}:")
            print(f"Período: {serie.index.min()} a {serie.index.max()}")
            print(f"Total de observações: {len(serie)}\n")
            
            dados_fluxo[nome] = serie
        
        except Exception as e:
            print(f"Erro: {e}")
    
    df_fluxo_capitais = pd.concat(dados_fluxo.values(), axis=1)
    return df_fluxo_capitais

df_bacen = coletar_fluxo_capitais()


# todo futuro. -> caso vença
# %%
df_bacen.columns = [
    'meta_taxa_selic',
    'taxa_selic',
    'taxa_cdi',
    'dolar_cambio_livre_p_tax',
    'euro_cambio_livre',
    'iene_cambio_livre'
]
df_bacen.tail(10)
df_bacen.head()
df_bacen.to_csv('../data/bronze/dados_bacen.csv', index=True)
# %%
from bcb import Expectativas

em = Expectativas()
ep = em.get_endpoint('ExpectativasMercadoTop5Anuais')# %%

# %%

df = (
    ep.query()
    .filter(ep.Indicador == "Selic")
    .select(
        ep.Data,
        ep.DataReferencia,
        ep.Mediana
    )
    .collect()
)

df["Data"] = pd.to_datetime(df["Data"])
df["DataReferencia"] = pd.to_numeric(
    df["DataReferencia"],
    errors="coerce"
)

df = df[
    (df["Data"].dt.year >= 2000) &
    (df["DataReferencia"] == df["Data"].dt.year + 1)
]

df = df.rename(columns={
    "Data": "data",
    "Mediana": "expectativa_selic_1y"
})

df = df[
    ["data", "expectativa_selic_1y"]
].sort_values("data")

# Remove registros exatamente iguais
df = df.drop_duplicates(
    subset=["data", "expectativa_selic_1y"],
    keep="first"
)

duplicados = (
    df[df["data"].duplicated(keep=False)]
    .sort_values("data")
)

print(duplicados.head(50))

print(
    df[df["data"].duplicated(keep=False)]
    .sort_values("data")
    .head(50)
)


df = (
    df.groupby("data", as_index=False)["expectativa_selic_1y"]
      .mean()
      .sort_values("data")
)

df.to_csv(
    "../data/bronze/expectativas.csv",
    index=False
)


# %%
print(df[df["data"].duplicated(keep=False)].head(30))
# %%
