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

#variaveis dependentes (ibovespa_br_returns, 
# dolar_cambio_livre_p_tax_zscore, risco_brasil_zscore, 
# vix_zscore, petro_brent_zscore)
features = [
        'ibovespa_br_returns', 'ibovespa_br_momentum', 'ibovespa_br_zscore',
        'vix_zscore', 'petro_brent_zscore', 'risco_brasil_zscore',
        'shanghai_china_zscore', 's&p500_eua_zscore',
        'inflacao_mensal_pct_change_lag_1m', 'taxa_selic_zscore',
        'dolar_cambio_livre_p_tax_zscore', 'euro_cambio_livre_zscore', #'expectativa_selic_1y_zscore'
]

arquivo = BASE_DIR / 'data/gold/macro_features_hmm.parquet'
print(f"Arquivo Gold: {arquivo}")
df_features = build_feature_store(arquivo, features, start_date='2002-08-31', end_date='2026-03-01', name_feature_store='features_model')


tickers = [

# Bancos & Serviços Financeiros

'ITUB4.SA',

'ITSA4.SA',

'BBDC4.SA',

'BBDC3.SA',

'BBAS3.SA',

'BPAC11.SA',

'SANB11.SA',

'ABCB4.SA',

'BPAN4.SA',

'B3SA3.SA',

'BBSE3.SA',

'CXSE3.SA',

'PSSA3.SA',

'IRBR3.SA',

# Petróleo, Gás & Petroquímica

'PETR4.SA',

'PETR3.SA',

'PRIO3.SA',

'BRAV3.SA', # ex-RRRP3 / 3R + Enauta

'RECV3.SA',

'ENEV3.SA',

'VBBR3.SA', # Vibra

'UGPA3.SA', # Ultrapar

'BRKM5.SA',

'UNIP6.SA',

# Mineração, Siderurgia & Papel/Celulose

'VALE3.SA',

'CSNA3.SA',

'USIM5.SA',

'GGBR4.SA',

'GOAU4.SA',

'CMIN3.SA',

'BRAP4.SA',

'AURA33.SA',

'SUZB3.SA',

'KLBN11.SA',

'KLBN4.SA',

# Energia Elétrica & Saneamento

'EQTL3.SA',

'CMIG4.SA',

'TAEE11.SA',

'CPFE3.SA',

'EGIE3.SA',

'SBSP3.SA',

'CSMG3.SA',

'SAPR11.SA', # Sanepar

'CPLE6.SA',

'CPLE3.SA',

'TRPL4.SA',

'ISAE4.SA',

'ALUP11.SA',

'ELET3.SA',

'ELET6.SA',

'LIGT3.SA',

# Varejo, Consumo & Saúde

'ABEV3.SA',

'RADL3.SA', # Raia Drogasil

'ASAI3.SA', # Assaí

'CRFB3.SA', # Carrefour

'NTCO3.SA', # Natura

'HYPE3.SA', # Hypera

'LREN3.SA', 'VIVA3.SA', 'MGLU3.SA', 'BHIA3.SA', 'AMER3.SA', 'PCAR3.SA', 'LJQQ3.SA', 'CVCB3.SA', 'AZZA3.SA', 'SMFT3.SA', 'RDOR3.SA', 'HAPV3.SA', 'FLRY3.SA',

# Bens de Capital, Indústria & Tecnologia

'WEGE3.SA', 'EMBR3.SA', 'TOTS3.SA', 'POSI3.SA', 'LWSA3.SA', 'INTB3.SA', 'KEPL3.SA', 'FRAS3.SA', 'ROMI3.SA', 'POMO4.SA',  'GGPS3.SA', 'STBP3.SA',

# Logística & Transporte

'RAIL3.SA', 'ECOR3.SA', 'CCRO3.SA', 'RENT3.SA', 'MOVI3.SA', 'AZUL4.SA', 'GOLL4.SA',

# Imobiliário & Construção

'MULT3.SA', 'ALOS3.SA', 'IGTI11.SA', 'CYRE3.SA', 'MRVE3.SA', 'EZTC3.SA', 'MDNE3.SA', 'CURY3.SA', 'DIRR3.SA', 'CSUD3.SA',

# Agronegócio & Proteína
'JBSS3.SA','BRFS3.SA','MRFG3.SA','SLCE3.SA', 'SMTO3.SA', 'AGRO3.SA','SOJA3.SA',

# Educação & Outros
'COGN3.SA', 'YDUQ3.SA', 'ANIM3.SA', 'OIBR3.SA'
]

tickers_unicos = list(dict.fromkeys(tickers))

#caminho = BASE_DIR / 'data/stocks/stocks_segmentos.csv'
#df = pd.read_csv(caminho, sep=',')
#tickers = df['symbol']
#carteiras = []
#for a in tickers:
#    if len(a) <= 5:
#        carteiras.append(a+'.SA')


# %%
retornos_sinal, retornos_execucao, ativos_risco, colunas_operacao = pipeline_returns(tickers_unicos, BASE_DIR)

datas_comuns = df_features.index.intersection(retornos_sinal.index).intersection(retornos_execucao.index)
df_features = df_features.loc[datas_comuns].sort_index()
retornos_sinal = retornos_sinal.loc[datas_comuns].sort_index()
retornos_execucao = retornos_execucao.loc[datas_comuns].sort_index()


# %%

df_rodadas, df_resultado ,df_pesos_historicos, df_carteiras, df_metricas_resumo = run_walk_forward_motor(
    df_features=df_features, 
    retornos_sinal=retornos_sinal,         # Usado no Treino / HMM / Bellman / Scores
    retornos_execucao=retornos_execucao,   # Usado estritamente no Teste Diário físico
    acoes_disponiveis=ativos_risco, 
    colunas_hmm=df_features.columns, 
    colunas_operacao=colunas_operacao,
    ano_inicio_operacao=2015, 
    janela_teste=1, 
    metrica_otimizacao='adaptativo',
    anos_memoria_treino=4, 
    tempo_regime=21, 
    limite_max_por_ativo=0.10,
    limite_min_por_ativo = 0.03,
    custo_corretagem=0.005,             
    custo_slippage=0.005,              
    capital_inicial=100000,
    numero_ativos=20,
    matriz_transicao="fixa",
    alpha_ema=0.20,
    margem_troca=0.20
)
# %%
vis.plot_full_history(df_resultado, ano_inicio=2010)
vis.plot_regimes_historicos(df_resultado)
vis.plot_heatmap_alpha_mensal(df_resultado)
vis.analise_comparativa_benchmark_flag(df_resultado, df_features,flag_vline=1)
vis.plot_scatter_retornos_mensais(df_resultado, coluna_bench='Retorno_Benchmark_Hibrido_Dinamico')
vis.plot_distribuicao_retornos(df_resultado)
vis.analise_atribuicao_alpha(df_resultado)
vis.plot_evolucao_alocacao(df_resultado)
vis.plot_retorno_anual_comparativo(df_resultado)
vis.plot_dashboard_rodadas(df_rodadas)

df_resultado.to_csv('df_resultados.csv')
df_rodadas.to_csv('df_rodadas.csv')
df_carteiras.to_csv('df_carteiras.csv')
df_pesos_historicos.to_csv('df_pesos.csv')

# %%
import plotly.graph_objects as go
import pandas as pd

def plot_evolucao_exposicao_com_regimes(df_resultado):
    """
    Gera um gráfico dinâmico de área empilhada (Ações vs CDI)
    com faixas coloridas no fundo representando os Regimes do HMM.
    """
    df_plot = df_resultado.copy()
    
    # Garante a escala de 0 a 100% para o gráfico
    df_plot['Exposicao_Acoes_Pct'] = df_plot['Exposicao_Acoes'] * 100
    df_plot['Exposicao_CDI_Pct'] = 100 - df_plot['Exposicao_Acoes_Pct']
    
    fig = go.Figure()
    
    # 1. Camada de Área Empilhada: Caixa (CDI)
    fig.add_trace(go.Scatter(
        x=df_plot.index, 
        y=df_plot['Exposicao_CDI_Pct'],
        mode='lines',
        name='Caixa (CDI)',
        stackgroup='one',
        groupnorm='percent',
        marker_color='rgba(230, 230, 230, 0.5)', # Cinza neutro
        line=dict(width=0.5)
    ))
    
    # 2. Camada de Área Empilhada: Exposição em Ações
    fig.add_trace(go.Scatter(
        x=df_plot.index, 
        y=df_plot['Exposicao_Acoes_Pct'],
        mode='lines',
        name='Exposição em Ações',
        stackgroup='one',
        marker_color='rgba(31, 119, 180, 0.85)', # Azul institucional contínuo
        line=dict(width=1)
    ))

    # =========================================================================
    # --- MAPEAMENTO DOS REGIMES DO HMM (FAIXAS VERTICAIS DE BACKGROUND) ---
    # =========================================================================
    # Configuração de cores suaves (transparentes) para o fundo não cobrir os dados
    cores_regimes = {
        0: 'rgba(46, 204, 113, 0.08)',   # Verde muito suave (Bull_Baixa_Vol)
        1: 'rgba(52, 152, 219, 0.05)',   # Azul muito suave (Transicao_Normal)
        2: 'rgba(241, 196, 15, 0.08)',   # Amarelo muito suave (Correcao)
        3: 'rgba(231, 76, 60, 0.12)'     # Vermelho visível (Crise_Panico)
    }
    
    # Identifica os pontos exatos onde o regime macro mudou na história
    df_plot['Mudou_Regime'] = df_plot['Regime_Macro'].diff().fillna(0) != 0
    datas_mudanca = df_plot[df_plot['Mudou_Regime']].index.tolist()
    
    # Garante os limites inicial e final da série histórica
    datas_limite = [df_plot.index[0]] + datas_mudanca + [df_plot.index[-1]]
    
    # Desenha as caixas (shapes) verticais no layout do Plotly
    shapes = []
    for idx in range(len(datas_limite) - 1):
        data_ini = datas_limite[idx]
        data_fim = datas_limite[idx + 1]
        
        # Pega o regime vigente naquele intervalo de tempo
        regime_vigente = df_plot.loc[data_ini, 'Regime_Macro']
        
        shapes.append(dict(
            type="rect",
            xref="x",
            yref="paper", # Trava o topo e o fundo do shape na moldura do gráfico
            x0=data_ini,
            y0=0,
            x1=data_fim,
            y1=1,
            fillcolor=cores_regimes.get(regime_vigente, 'rgba(0,0,0,0)'),
            line=dict(width=0), # Sem bordas para não poluir
            layer="below" # Força as faixas coloridas a ficarem ATRÁS das áreas empilhadas
        ))
        
    # Adiciona traços invisíveis na legenda apenas para mapear a cor de cada regime pro usuário
    nomes_regimes = {0: "Bull Market", 1: "Transição", 2: "Correção", 3: "Crise/Pânico"}
    for reg, cor in cores_regimes.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=cor.replace('0.', '0.9'), symbol='square'), # Cor opaca na legenda
            name=f"Regime: {nomes_regimes[reg]}",
            showlegend=True
        ))

    # Configurações estéticas e eixos
    fig.update_layout(
        title='<b>Evolução da Alocação Dinâmica vs Regimes do HMM</b><br><sup>Rotação tática sob estresse: fundos coloridos indicam o regime de mercado determinado pelo modelo</sup>',
        xaxis_title='Tempo',
        yaxis_title='Alocação do Capital (%)',
        hovermode='x unified',
        template='plotly_white',
        shapes=shapes, # Injeta os backgrounds calculados
        yaxis=dict(ticksuffix='%', range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5) # Legenda na parte inferior interna
    )
    
    fig.show()

# Chamada da nova função
plot_evolucao_exposicao_com_regimes(df_resultado)



# %%
for regime, grupo in df_resultado.groupby("Nome_Regime"):

    ret_modelo = (1 + grupo["Retorno_Modelo"]).prod() - 1
    ret_ibov = (1 + grupo["Retorno_Benchmark_Hibrido_Dinamico"]).prod() - 1

    alpha = ret_modelo - ret_ibov

    print(f"{regime}: Alpha = {alpha * 100:.2f}%")


def analisar_regimes_out_of_sample(df_resultado):

    resumo = (
        df_resultado
        .groupby("Regime_Macro")
        .agg({
            "Retorno_Benchmark":"mean",
            "Retorno_Modelo":"mean",
            "Exposicao_Acoes":"mean"
        })
    )

    return resumo

resumo_regimes = analisar_regimes_out_of_sample(df_resultado)
print(resumo_regimes)


# %%
import pandas as pd
import numpy as np

df_resultado.index = pd.to_datetime(df_resultado.index)
df_resultado = df_resultado.sort_index()

# 2. Cálculo do CAGR Total (Mapeamento geométrico exato)
patrimonio_inicial = df_resultado['Patrimonio'].iloc[0]
patrimonio_final = df_resultado['Patrimonio'].iloc[-1]
dias_totais = len(df_resultado)
anos = dias_totais / 252.0
cagr_total = (patrimonio_final / patrimonio_inicial) ** (1 / anos) - 1

# 3. Métricas baseadas na série de retornos diários
retornos = df_resultado['Retorno_Modelo'].dropna()

# Volatilidade Anualizada
vol_anualizada = retornos.std() * np.sqrt(252)

# CORREÇÃO: Retorno Anualizado Composto (Alinhado ao padrão de fundos)
retorno_anualizado_comp = (1 + retornos.mean()) ** 252 - 1
taxa_livre_risco = 0.0

# Sharpe Ratio Corrigido
sharpe_ratio = (retorno_anualizado_comp - taxa_livre_risco) / vol_anualizada

# --- NOVAS MÉTRICAS DE RISCO ASYMMETRIC ---

# CORREÇÃO: Downside Deviation correto (raiz da média dos quadrados dos retornos abaixo de zero)
target_return = 0.0
retornos_abaixo_target = retornos[retornos < target_return]
downside_deviation = np.sqrt(np.mean(retornos_abaixo_target ** 2)) * np.sqrt(252)

# Sortino Ratio Corrigido
sortino_ratio = (retorno_anualizado_comp - taxa_livre_risco) / downside_deviation if downside_deviation != 0 else np.nan

# 4. Max Drawdown
rolling_max = df_resultado['Patrimonio'].cummax()
drawdown = (df_resultado['Patrimonio'] - rolling_max) / rolling_max
max_drawdown = drawdown.min()

# Calmar Ratio
calmar_ratio = cagr_total / abs(max_drawdown) if max_drawdown != 0 else np.nan

# Tail Ratio (Percentil 95 / Valor absoluto do Percentil 5)
p95 = np.percentile(retornos, 95)
p05 = abs(np.percentile(retornos, 5))
tail_ratio = p95 / p05 if p05 != 0 else np.nan

# --- IMPLEMENTAÇÃO: VaR e CVaR HISTÓRICO ---

# Nível de significância alfa = 5% (Confiança 95%)
var_95_diario = np.percentile(retornos, 5) # Retorno do pior 5º percentil
cvar_95_diario = retornos[retornos <= var_95_diario].mean() # Média dos 5% piores retornos

# Nível de significância alfa = 1% (Confiança 99% - Estresse de Cauda Externa)
var_99_diario = np.percentile(retornos, 1)
cvar_99_diario = retornos[retornos <= var_99_diario].mean()

# 5. Consolidando em um DataFrame Resumo Ampliado com Métricas de Cauda
tabela_resumo = pd.DataFrame({
    "Métrica de Risco / Retorno": [
        "CAGR Total (Geométrico)", 
        "Volatilidade Anualizada", 
        "Sharpe Ratio",
        "Sortino Ratio",
        "Máximo Drawdown (Max DD)",
        "Calmar Ratio",
        "Tail Ratio (95/5)",
        "VaR Histórico Diário (95%)",
        "CVaR Histórico Diário (95%)",
        "VaR Histórico Diário (99%)",
        "CVaR Histórico Diário (99%)"
    ],
    "Estratégia HMM + Bellman": [
        f"{cagr_total * 100:.2f}%",
        f"{vol_anualizada * 100:.2f}%",
        f"{sharpe_ratio:.2f}",
        f"{sortino_ratio:.2f}",
        f"{max_drawdown * 100:.2f}%",
        f"{calmar_ratio:.2f}",
        f"{tail_ratio:.2f}",
        f"{var_95_diario * 100:.2f}%",
        f"{cvar_95_diario * 100:.2f}%",
        f"{var_99_diario * 100:.2f}%",
        f"{cvar_99_diario * 100:.2f}%"
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