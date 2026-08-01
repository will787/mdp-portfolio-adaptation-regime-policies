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
import src.visualization.plots as vis
from src.pipeline.returns import pipeline_returns
from src.pipeline.selecao_features import build_feature_store
from src.utils.read_dir import try_read_dir

BASE_DIR = try_read_dir()

features = [
        'ibovespa_br_returns', 'ibovespa_br_volatily', 'ibovespa_br_momentum', 'ibovespa_br_zscore',
        'vix_zscore', 'petro_brent_zscore', 'taxa_selic_zscore', 'risco_brasil_zscore',  #'meta_taxa_selic_zscore',
        'shanghai_china_returns',  'inflacao_mensal_pct_change_lag_1m'
        #'dolar_cambio_livre_p_tax_zscore', 'euro_cambio_livre_zscore', 'iene_cambio_livre_zscore'
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
    'B3SA3.SA','RENT3.SA','MOVI3.SA','SLCE3.SA','SMTO3.SA','CMIN3.SA','YDUQ3.SA','CPLE6.SA','BRKM5.SA','LWSA3.SA',
    'GOLL4.SA', 'ENBR3.SA', 'ALUP11.SA', 'CSMG3.SA', 'BRAP4.SA', 'AGRO3.SA', 'BPAN4.SA', 'SOJA3.SA',
    'CPLE3.SA', 'TRPL4.SA', 'IGTI11.SA', 'ARZZ3.SA', 'LIGT3.SA',
    'GOAU4.SA','ELET3.SA','ELET6.SA','ISAE4.SA'
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
    tempo_regime=22, 
    limite_max_por_ativo=0.08,
    custo_corretagem=0.0005,             
    custo_slippage=0.001,              
    capital_inicial=100000
)

# %%

vis.plot_full_history(df_resultado, ano_inicio=2010)
vis.plot_regimes_historicos(df_resultado)
vis.plot_heatmap_alpha_mensal(df_resultado)
vis.plot_distribuicao_retornos(df_resultado)
vis.analise_atribuicao_alpha(df_resultado)
vis.plot_evolucao_alocacao(df_resultado)
vis.plot_retorno_anual_comparativo(df_resultado)
vis.plot_dashboard_rodadas(df_rodadas)

df_resultado.to_csv('df_resultados.csv')
df_rodadas.to_csv('df_rodadas.csv')
df_recompensas.to_csv('df_recompensas.csv')
df_carteiras.to_csv('df_carteiras.csv')
df_pesos.to_csv('df_pesos.csv')
# %%
import pandas as pd
import numpy as np
df_resultado.index = pd.to_datetime(df_resultado.index)
df_resultado = df_resultado.sort_index()

# 2. Cálculo do CAGR Total (início ao fim do backtest)
patrimonio_inicial = df_resultado['Patrimonio'].iloc[0]
patrimonio_final = df_resultado['Patrimonio'].iloc[-1]
dias_totais = len(df_resultado)
anos = dias_totais / 252.0
cagr_total = (patrimonio_final / patrimonio_inicial) ** (1 / anos) - 1

# 3. Métricas baseadas na série de retornos diários ('Retorno_Modelo')
retornos = df_resultado['Retorno_Modelo'].dropna()

# Volatilidade Anualizada
vol_anualizada = retornos.std() * np.sqrt(252)

# Retorno Médio Anualizado
retorno_medio_anualizado = retornos.mean() * 252

# Sharpe Ratio
taxa_livre_risco = 0.0
sharpe_ratio = (retorno_medio_anualizado - taxa_livre_risco) / vol_anualizada

# --- NOVAS MÉTRICAS ---

# Downside Deviation (apenas retornos abaixo de zero)
retornos_negativos = retornos[retornos < 0]
downside_deviation = retornos_negativos.std() * np.sqrt(252)

# Sortino Ratio
sortino_ratio = (retorno_medio_anualizado - taxa_livre_risco) / downside_deviation if downside_deviation != 0 else np.nan

# 4. Max Drawdown
rolling_max = df_resultado['Patrimonio'].cummax()
drawdown = (df_resultado['Patrimonio'] - rolling_max) / rolling_max
max_drawdown = drawdown.min()

# Calmar Ratio (usando o CAGR Total absoluto)
calmar_ratio = cagr_total / abs(max_drawdown) if max_drawdown != 0 else np.nan

# Tail Ratio (Percentil 95 / Valor absoluto do Percentil 5)
p95 = np.percentile(retornos, 95)
p05 = abs(np.percentile(retornos, 5))
tail_ratio = p95 / p05 if p05 != 0 else np.nan

# 5. Consolidando em um DataFrame Resumo Ampliado
tabela_resumo = pd.DataFrame({
    "Métrica de Risco / Retorno": [
        "CAGR Total", 
        "Volatilidade Anualizada", 
        "Sharpe Ratio",
        "Sortino Ratio",
        "Máximo Drawdown (Max DD)",
        "Calmar Ratio",
        "Tail Ratio (95/5)"
    ],
    "Estratégia HMM + Bellman": [
        f"{cagr_total * 100:.2f}%",
        f"{vol_anualizada * 100:.2f}%",
        f"{sharpe_ratio:.2f}",
        f"{sortino_ratio:.2f}",
        f"{max_drawdown * 100:.2f}%",
        f"{calmar_ratio:.2f}",
        f"{tail_ratio:.2f}"
    ]
})

print(tabela_resumo.to_string(index=False))

# %%
import plotly.express as px
import plotly.graph_objects as go


def plot_evolucao_exposicao(df_resultado):
    """
    Gera um gráfico dinâmico de área empilhada mostrando a evolução 
    da exposição tática do modelo (Ações vs CDI) ao longo do tempo.
    """
    df_plot = df_resultado.copy()
    
    # Garante que a coluna está no formato correto (escala 0 a 100% para o gráfico)
    df_plot['Exposicao_Acoes_Pct'] = df_plot['Exposicao_Acoes'] * 100
    df_plot['Exposicao_CDI_Pct'] = 100 - df_plot['Exposicao_Acoes_Pct']
    
    fig = go.Figure()
    
    # Camada 1: Caixa (CDI) - Base de proteção do portfólio
    fig.add_trace(go.Scatter(
        x=df_plot.index, 
        y=df_plot['Exposicao_CDI_Pct'],
        mode='lines',
        name='Caixa (CDI)',
        stackgroup='one', # Ativa o empilhamento automático
        groupnorm='percent', # Força a soma das áreas a travar em 100%
        marker_color='rgba(220, 220, 220, 0.6)', # Cinza claro neutro
        line=dict(width=0.5)
    ))
    
    # Camada 2: Alocação em Risco (Ibovespa)
    fig.add_trace(go.Scatter(
        x=df_plot.index, 
        y=df_plot['Exposicao_Acoes_Pct'],
        mode='lines',
        name='Exposição em Ações',
        stackgroup='one',
        marker_color='rgba(30, 144, 255, 0.8)', # Azul institucional vibrante
        line=dict(width=1)
    ))
    
    # Configurações estéticas e eixos
    fig.update_layout(
        title='<b>Evolução da Alocação Dinâmica do Portfólio</b><br><sup>Rotação tática de ativos ditada pela Equação de Bellman</sup>',
        xaxis_title='Tempo',
        yaxis_title='Alocação do Capital (%)',
        hovermode='x unified',
        template='plotly_white',
        yaxis=dict(ticksuffix='%', range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.show()

plot_evolucao_exposicao(df_resultado)

# %%
