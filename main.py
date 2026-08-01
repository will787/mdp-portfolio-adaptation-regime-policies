# %%
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
print(Path.cwd())
print(sys.path[0])

try:
    base = Path(__file__).resolve().parent
except NameError:
    base = Path.cwd()

while not (base / "src").exists():
    if base.parent == base:
        raise RuntimeError("Não encontrei a pasta 'src'.")
    base = base.parent

if str(base) not in sys.path:
    sys.path.insert(0, str(base))

print("Projeto:", base)
# %%


from src.engine.motor import run_walk_forward_motor
from src.pipeline.returns import pipeline_returns
from src.pipeline.selecao_features import build_feature_store
from src.utils.read_dir import try_read_dir

BASE_DIR = try_read_dir()

features = [
        'ibovespa_br_returns', 'ibovespa_br_volatily', 'ibovespa_br_momentum', 'ibovespa_br_zscore',
        'vix_zscore', 'petro_brent_zscore', 'taxa_selic_zscore', 'risco_brasil_zscore', 
        's&p500_eua_returns', 'shanghai_china_returns',  'inflacao_mensal_pct_change_lag_1m', 'meta_taxa_selic_zscore',
        'dolar_cambio_livre_p_tax_zscore', 'euro_cambio_livre_zscore', 'iene_cambio_livre_zscore'
]

arquivo = BASE_DIR / 'data/gold/macro_features_hmm.parquet'
print(f"Arquivo Gold: {arquivo}")
df_features = build_feature_store(arquivo, features, start_date='2002-08-31', end_date='2026-03-01', name_feature_store='features_model')

tickers = [
    'ITUB4.SA', 'ITSA4.SA', 'BBDC4.SA', 'BBDC3.SA','BBAS3.SA', 'BPAC11.SA', 'SANB11.SA', 'ABCB4.SA',
    'PETR4.SA', 'PETR3.SA', 'PRIO3.SA','RRRP3.SA', 'RECV3.SA','ENEV3.SA',
    'VALE3.SA','CSNA3.SA','USIM5.SA','GGBR4.SA', 'AURA33.SA',
    'SUZB3.SA','KLBN11.SA','KLBN4.SA',
    'EQTL3.SA','CMIG4.SA','TAEE11.SA','CPFE3.SA','EGIE3.SA',
    'SBSP3.SA','RAIL3.SA','ECOR3.SA','CCRO3.SA',
    'ABEV3.SA','LREN3.SA','VIVA3.SA', 'MGLU3.SA','BHIA3.SA','AMER3.SA','PCAR3.SA',
    'RDOR3.SA','HAPV3.SA','FLRY3.SA', 'TOTS3.SA','POSI3.SA', 'MDNE3.SA',
    'WEGE3.SA','EMBR3.SA','BBSE3.SA','CXSE3.SA','PSSA3.SA',
    'VIVT3.SA','TIMS3.SA', 'MULT3.SA','ALOS3.SA',
    'CYRE3.SA','MRVE3.SA','EZTC3.SA', 'JBSS3.SA','BRFS3.SA','MRFG3.SA',
    'IRBR3.SA','COGN3.SA','OIBR3.SA','GOLL4.SA', 'LJQQ3.SA', 'ANIM3.SA', 'AZUL4.SA',
    'KEPL3.SA', 'FRAS3.SA', 'ROMI3.SA', 'UNIP6.SA', 'CSUD3.SA', 'CVCB3.SA', 'VIIA3.SA',
    'B3SA3.SA','RENT3.SA','MOVI3.SA','SLCE3.SA','SMTO3.SA','CMIN3.SA','YDUQ3.SA','CPLE6.SA','BRKM5.SA','LWSA3.SA'
    ,'GOLL4.SA', 'ENBR3.SA', 'ALUP11.SA', 'CSMG3.SA', 'BRAP4.SA', 'AGRO3.SA', 'BPAN4.SA', 'SOJA3.SA',
    'CPLE3.SA', 'TRPL4.SA', 'IGTI11.SA', 'ARZZ3.SA', 'LIGT3.SA'
]

tickers_unicos = list(dict.fromkeys(tickers))
retornos_sinal, retornos_execucao, ativos_risco, colunas_operacao = pipeline_returns(tickers_unicos, BASE_DIR)

datas_comuns = df_features.index.intersection(retornos_sinal.index).intersection(retornos_execucao.index)
df_features = df_features.loc[datas_comuns].sort_index()
retornos_sinal = retornos_sinal.loc[datas_comuns].sort_index()
retornos_execucao = retornos_execucao.loc[datas_comuns].sort_index()

df_resultado, df_rodadas, df_recompensas, df_carteiras, df_pesos = run_walk_forward_motor(
    df_features=df_features, 
    retornos_sinal=retornos_sinal,         # Usado no Treino / HMM / Bellman / Scores
    retornos_execucao=retornos_execucao,   # Usado estritamente no Teste Diário físico
    acoes_disponiveis=ativos_risco, 
    colunas_hmm=df_features.columns, 
    colunas_operacao=colunas_operacao,
    ano_inicio_operacao=2010, 
    janela_teste=1, 
    metrica_otimizacao='adaptativo',
    anos_memoria_treino=5, 
    tempo_regime=42, 
    limite_max_por_ativo=0.08,
    custo_corretagem=0.0005,             
    custo_slippage=0.001,              
    capital_inicial=100000
)

# %%
