
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
    """Módulo 2: Prepara as arenas de Sinais e Execução de forma limpa e estável"""
    print("2. Construindo Pipeline de Retornos Operacionais...")
    import yfinance as yf
    
    caminho_selic = BASE_DIR / 'data/bronze/dados_bacen.csv'
    dados_completos = yf.download(tickers, start="2002-08-31", end="2026-06-01", auto_adjust=False)

    if dados_completos.index.tz is not None:
        dados_completos.index = dados_completos.index.tz_localize(None)
    dados_completos.index.name = "data"

    dados_adj = dados_completos["Adj Close"].ffill()
    retornos_sinal = dados_adj.pct_change().fillna(0.0) # Seguro: Retorno 0.0 mantém a média estável

    retornos_execucao = retornos_sinal.copy()
    
    df_bacen = pd.read_csv(caminho_selic, index_col="data", parse_dates=True)
    selic_diaria_limpa = (df_bacen['taxa_selic'] / 100.0).ffill().fillna(0.0)
    cdi_alinhado = selic_diaria_limpa.reindex(retornos_sinal.index).ffill().fillna(0.0)
    
    retornos_sinal["CDI"] = cdi_alinhado
    retornos_execucao["CDI"] = cdi_alinhado

    return retornos_sinal, retornos_execucao, dados_adj.columns.tolist(), dados_adj.columns.tolist() + ['CDI']



def pipeline_returns_xx(tickers, BASE_DIR):

    caminho_selic = BASE_DIR / 'data/bronze/dados_bacen.csv'
    caminho_cadastro = BASE_DIR / 'data/stocks/tickers_setores.csv'

    # ==========================================================
    # 1. CADASTRO
    # ==========================================================

    cadastro_ativos = pd.read_csv(caminho_cadastro)

    cadastro_ativos["Ticker"] = (
        cadastro_ativos["Ticker"]
        .astype(str)
        .str.strip()
    )

    # ==========================================================
    # 2. NORMALIZAÇÃO DOS TICKERS
    # ==========================================================

    tickers = list(dict.fromkeys(tickers))

    # remove .SA para conseguir cruzar com o cadastro B3
    tickers_b3 = [ticker.replace(".SA", "") for ticker in tickers]

    cadastro_ativos = cadastro_ativos[
        cadastro_ativos["Ticker"].isin(tickers_b3)
    ].copy()

    # ==========================================================
    # 3. DOWNLOAD
    # ==========================================================

    dados_completos = yf.download(
        tickers,
        start="2002-08-31",
        end="2026-06-01",
        auto_adjust=False
    )

    if dados_completos.index.tz is not None:
        dados_completos.index = dados_completos.index.tz_localize(None)

    dados_completos.index.name = "data"

    # ==========================================================
    # 4. PREÇOS
    # ==========================================================

    dados_adj = dados_completos["Adj Close"].copy()
    dados_close = dados_completos["Close"].copy()

    # ==========================================================
    # 5. DISPONIBILIDADE
    # ==========================================================

    disponibilidade = dados_adj.notna()
    disponibilidade.index.name = "data"

    # ==========================================================
    # 6. RETORNOS
    # ==========================================================

    retornos_sinal = dados_adj.pct_change()
    retornos_sinal.index.name = "data"

    retornos_execucao = dados_close.pct_change()
    retornos_execucao.index.name = "data"

    # ==========================================================
    # 7. CDI
    # ==========================================================

    df_bacen = pd.read_csv(
        caminho_selic,
        index_col="data",
        parse_dates=True
    )

    selic_diaria_limpa = (
        df_bacen["taxa_selic"]
        .div(100)
        .ffill()
        .fillna(0.0)
    )

    cdi_alinhado = (
        selic_diaria_limpa
        .reindex(retornos_sinal.index)
        .ffill()
        .fillna(0.0)
    )

    retornos_sinal["CDI"] = cdi_alinhado
    retornos_execucao["CDI"] = cdi_alinhado

    disponibilidade['CDI'] = True

    # ==========================================================
    # 8. UNIVERSO DE RISCO
    # ==========================================================

    ativos_risco = [
        ticker
        for ticker in tickers
        if ticker in dados_adj.columns
    ]

    colunas_operacao = ativos_risco + ["CDI"]

    # ==========================================================
    # 9. RETORNO
    # ==========================================================

    return (
        retornos_sinal,
        retornos_execucao,
        disponibilidade,
        ativos_risco,
        colunas_operacao,
        cadastro_ativos
    )