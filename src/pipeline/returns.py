
from pathlib import Path
import os
import sys
import json 

import pandas as pd
import yfinance as yf

ROOT = Path.cwd()
print(ROOT)

sys.path.insert(0, str(ROOT))


def pipeline_returns(tickers, BASE_DIR):
    """Módulo 2: Prepara as arenas de Sinais e Execução (Ações + Caixa)"""

    print("2. Construindo Pipeline de Retornos Operacionais...")
    caminho_selic = BASE_DIR / 'data/bronze/dados_bacen.csv'
    print(f"Arquivo Selic: {caminho_selic}")
    dados_completos = yf.download(
        tickers,
        start="2002-08-31",
        end="2026-06-01",
        auto_adjust=False
    )

    if dados_completos.index.tz is not None:
        dados_completos.index = dados_completos.index.tz_localize(None)
    dados_completos.index.name = "data"

    dados_adj = dados_completos["Adj Close"].ffill()
    retornos_sinal = dados_adj.pct_change().fillna(0.0)
    retornos_sinal.index.name = "data"

    dados_close = dados_completos["Adj Close"].ffill()
    retornos_execucao = dados_close.pct_change().fillna(0.0)
    retornos_execucao.index.name = "data"
    df_bacen = pd.read_csv(caminho_selic, index_col="data", parse_dates=True)
    selic_diaria_limpa = (df_bacen['taxa_selic'] / 100.0).ffill().fillna(0.0)

    cdi_alinhado = selic_diaria_limpa.reindex(retornos_sinal.index).ffill().fillna(0.0)
    retornos_sinal["CDI"] = cdi_alinhado
    retornos_execucao["CDI"] = cdi_alinhado

    ativos_risco = dados_adj.columns.tolist()
    colunas_operacao = ativos_risco + ['CDI']

    return retornos_sinal, retornos_execucao, ativos_risco, colunas_operacao
