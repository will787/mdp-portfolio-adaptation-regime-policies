# %%

import ipeadatapy as ipd
import pandas as pd 

def search_series(keyword):
    """Busca por séries relacionadas a um tema específico no IPEA Data. Dados sempre Macro"""
    df = ipd.list_series()
    ipd.list_series().columns
    ipd.metadata(big_theme='Macroeconômico', country='Brasil', frequency='Mensal')
    return df[df['NAME'].str.contains(keyword, case=False)]


inflacao = (search_series('Inflação'))
pib = (search_series('PIB'))
desemprego = (search_series('Desemprego'))

print(desemprego[['CODE', 'NAME']])
print(inflacao[['CODE', 'NAME']])
print(pib[['CODE', 'NAME']])

#ipd.timeseries('BM12_IPCAEXP1212') # inflacao
#ipd.timeseries('BM12_PIB12') #pib mensal
#ipd.timeseries('PNADC12_TDESOC12') # desemprego

series_monthly = {
    "pib_mensal": "BM12_PIB12",
    "desemprego_mensal": "PNADC12_TDESOC12",
    "inflacao_mensal": "BM12_IPCAEXP1212",
}

series_daily = {
    "risco_brasil": 'JPM366_EMBI366'
}

def get_monthly_series(series_dict, start_year):
    """
        Coleta séries mensais do IPEA Data a partir de um dicionário de séries e um ano de início.
        
        Parâmetros:
        series_dict (dict): Dicionário contendo os nomes das séries como chaves e os códigos das séries como valores.
        start_year (int): Ano de início para a coleta das séries.
        
        Retorna:
        pd.DataFrame: DataFrame contendo as séries coletadas.
    """
    data_frames = []
    start_year = start_year - 1
    for name, code in series_dict.items():
        try:
            series = ipd.timeseries(code, yearGreaterThan=start_year)
            df = series.reset_index()
            value_col = next(col for col in df.columns if col.startswith('VALUE'))
            df = df[['YEAR', 'MONTH', value_col]].copy()
            df = df.rename(columns={value_col: name})
            data_frames.append(df)
        except Exception as e:
            print(f"Erro ao coletar a série {name}: {e}")

    merged_df = data_frames[0]
    for df in data_frames[1:]:
        merged_df = pd.merge(merged_df, df, on=['YEAR', 'MONTH'], how='outer')

    merged_df = merged_df.sort_values(by=['YEAR', 'MONTH']).reset_index(drop=True)

    merged_df['data'] = pd.to_datetime(dict(year=merged_df['YEAR'], month=merged_df['MONTH'], day=1))

    merged_df = merged_df.set_index('data')

    return merged_df

def get_risco_brasil(start_year):
    """
        Coleta séries diárias do IPEA Data a partir de um dicionário de séries e um ano de início.
        
        Parâmetros:
        series_dict (dict): Dicionário contendo os nomes das séries como chaves e os códigos das séries como valores.
        start_year (int): Ano de início para a coleta das séries.
        
        Retorna:
        pd.DataFrame: DataFrame contendo as séries coletadas.
    """
    df = ipd.timeseries(
        "JPM366_EMBI366",
        yearGreaterThan=start_year
    )

    df = df.reset_index()

    value_col = next(
        c for c in df.columns
        if c.startswith("VALUE")
    )

    return (
        df[["DATE", value_col]]
        .rename(
            columns={
                "DATE":"data",
                value_col:"risco_brasil"
            }
        )
    )

# %%

cols = [
    'pib_mensal',
    'desemprego_mensal',
    'inflacao_mensal'
]

monthly = get_monthly_series(
    {
        "pib_mensal":"BM12_PIB12",
        "desemprego_mensal":"PNADC12_TDESOC12",
        "inflacao_mensal":"BM12_IPCAEXP1212"
    },
    start_year=2000
)

monthly = get_monthly_series(series_monthly, start_year=2000)
risco = get_risco_brasil(start_year=2000)

monthly.to_csv('../data/bronze/dados_ipea.csv', index=True)
risco.to_csv('../data/bronze/risco_brasil.csv', index=True)
# %%
