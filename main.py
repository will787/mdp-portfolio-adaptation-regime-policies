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
        'dolar_cambio_livre_p_tax_zscore', 'euro_cambio_livre_zscore', 'expectativa_selic_1y_zscore'
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

'LREN3.SA',

'VIVA3.SA',

'MGLU3.SA',

'BHIA3.SA',

'AMER3.SA',

'PCAR3.SA',

'LJQQ3.SA',

'CVCB3.SA',

'AZZA3.SA', # ex-ARZZ3 / Arezzo + Soma

'SMFT3.SA', # Smart Fit

'RDOR3.SA',

'HAPV3.SA',

'FLRY3.SA',

# Bens de Capital, Indústria & Tecnologia

'WEGE3.SA',

'EMBR3.SA',

'TOTS3.SA',

'POSI3.SA',

'LWSA3.SA',

'INTB3.SA', # Intelbras

'KEPL3.SA',

'FRAS3.SA',

'ROMI3.SA',

'POMO4.SA', # Marcopolo

'GGPS3.SA', # GPS

'STBP3.SA', # Santos Brasil

# Logística & Transporte

'RAIL3.SA',

'ECOR3.SA',

'CCRO3.SA',

'RENT3.SA',

'MOVI3.SA',

'AZUL4.SA',

'GOLL4.SA',

# Imobiliário & Construção

'MULT3.SA',

'ALOS3.SA',

'IGTI11.SA',

'CYRE3.SA',

'MRVE3.SA',

'EZTC3.SA',

'MDNE3.SA',

'CURY3.SA', # Cury

'DIRR3.SA', # Direcional

'CSUD3.SA',

# Agronegócio & Proteína

'JBSS3.SA',

'BRFS3.SA',

'MRFG3.SA',

'SLCE3.SA',

'SMTO3.SA',

'AGRO3.SA',

'SOJA3.SA',

# Educação & Outros

'COGN3.SA',

'YDUQ3.SA',

'ANIM3.SA',

'OIBR3.SA',

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
    ano_inicio_operacao=2010, 
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

print(df_rodadas[[
    'Rodada_Teste', 'Retorno_Teste_Modelo', 'Retorno_Teste_Ibov', 
    'Retorno_Teste_Bench_Dinamico', 'Alpha_Rodada', 'Sharpe', 'Max_Drawdown'
]])

# 2. Retorno total acumulado do modelo vs Benchmark
retorno_total_modelo = (1 + df_resultado['Retorno_Modelo']).prod() - 1
retorno_total_cdi = (1 + df_resultado['Retorno_Benchmark_Hibrido']).prod() - 1

print(f"\nResultado Acumulado (2015 - 2025):")
print(f"K-NESIAN HMM: {retorno_total_modelo:.2%}")
print(f"CDI Acumulado: {retorno_total_cdi:.2%}")
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

# %%

def analisar_resultados_por_regime(df):
    """
    Agrupa e calcula métricas institucionais de retorno, volatilidade, 
    eficiência e exposição física para cada regime predito pelo HMM.
    """
    # Garante ordenação e tratamento de nulos
    df = df.sort_index()
    
    # Lista para armazenar o dicionário de métricas de cada regime
    metricas_regimes = []
    
    # Identifica os regimes únicos presentes no backtest (ex: Estado 0, Estado 1...)
    regimes_unicos = df['Regime_Macro'].dropna().unique()
    
    for regime in regimes_unicos:
        # Filtra o DataFrame apenas para os dias em que o modelo esteve neste regime
        df_sub = df[df['Regime_Macro'] == regime]
        
        # Ignora se a amostra for insignificante (menos de 5 dias úteis)
        if len(df_sub) < 5:
            continue
            
        retornos = df_sub['Retorno_Modelo'].dropna()
        
        # 1. Contagem de dias e representatividade temporal
        total_dias = len(df_sub)
        porcentagem_tempo = (total_dias / len(df)) * 100
        
        # 2. Retorno Médio Anualizado Composto (Considerando a frequência do regime)
        # Usamos a média geométrica/composta diária trazida para a escala de 252 dias úteis
        retorno_medio_anual = (1 + retornos.mean()) ** 252 - 1
        
        # 3. Volatilidade Anualizada do Regime
        vol_anual = retornos.std() * np.sqrt(252)
        
        # 4. Sharpe Ratio do Regime (Taxa livre de risco considerada zero no subperíodo)
        sharpe = retorno_medio_anual / vol_anual if vol_anual != 0 else 0
        
        # 5. Exposição Média em Ações durante o Regime
        # Se você tiver a coluna de exposição direta no snapshot, use ela. 
        # Caso contrário, calculamos pela proporção do capital alocado em risco.
        if 'Exposicao' in df_sub.columns:
            exp_media = df_sub['Exposicao_Acoes'].mean()
        elif 'Capital_Acoes' in df_sub.columns and 'Patrimonio' in df_sub.columns:
            exp_media = (df_sub['Capital_Acoes'] / df_sub['Patrimonio']).mean()
        else:
            exp_media = np.nan
            
        # Consolida as métricas calculadas
        metricas_regimes.append({
            "Regime": int(regime),
            "Dias Ativo": total_dias,
            "% Tempo Fundo": f"{porcentagem_tempo:.1f}%",
            "Retorno Médio Anual": f"{retorno_medio_anual * 100:.2f}%",
            "Volatilidade Anual": f"{vol_anual * 100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Exposição Média": f"{exp_media * 100:.2f}%" if not np.isnan(exp_media) else "N/A"
        })
        
    # Transforma em DataFrame para exibição estruturada
    df_analise = pd.DataFrame(metricas_regimes).sort_values(by="Regime")
    
    print("\n" + "="*25 + " RAIO-X DE PERFORMANCE POR REGIME (HMM) " + "="*25)
    print(df_analise.to_string(index=False))
    print("="*90)
    
    return df_analise

df_regimes_summary = analisar_resultados_por_regime(df_resultado)
# %%

# %%

import numpy as np
import pandas as pd
from pathlib import Path

# 1. Definir o arquivo
arquivo = BASE_DIR / 'data/gold/macro_features_hmm.parquet'
print(f"Arquivo Gold: {arquivo}")

# 2. Ler as colunas reais existentes no Parquet
colunas_disponiveis = pd.read_parquet(arquivo).columns.tolist()

# 3. Lista de features desejadas
features_desejadas = [
    'ibovespa_br_returns',
    'ibovespa_br_momentum',
    'ibovespa_br_zscore',
    'vix_zscore',
    'petro_brent_zscore',
    'taxa_selic_zscore',
    'risco_brasil_zscore',
    'shanghai_china_pct_change_lag_1m',
    's&p500_eua_pct_change_lag_1m',
    'inflacao_mensal_pct_change_lag_1m',
    'dolar_cambio_livre_p_tax_zscore',
    'iene_cambio_livre_zscore',
]

# 4. Filtrar apenas as colunas que realmente existem no arquivo Parquet
features_validas = [f for f in features_desejadas if f in colunas_disponiveis]

print(
    f"✅ Features válidas carregadas no df_features ({len(features_validas)}):"
)
print(features_validas)

# 5. Carregar o Feature Store
df_features = build_feature_store(
    arquivo,
    features_validas,
    start_date='2002-08-31',
    end_date='2026-03-01',
    name_feature_store='features_model',
)

# 6. Mapear os grupos de teste APENAS com colunas que realmente existem em df_features
coluna_base = 'ibovespa_br_returns'

# Dicionário garantindo que a coluna_base está em TODOS os grupos
conjuntos_features = {
    # 1. Todas as variáveis válidas
    'Todas_Features': features_validas,
    # 2. Fatores Brasil + Ibov Returns
    'Fatores_Brasil': list(
        set([
            coluna_base,
            'taxa_selic_zscore',
            'risco_brasil_zscore',
            'inflacao_mensal_pct_change_lag_1m',
            'dolar_cambio_livre_p_tax_zscore',
        ])
        & set(features_validas)
    ),
    # 3. Fatores Globais + Ibov Returns
    'Fatores_Globais': list(
        set([
            coluna_base,
            'vix_zscore',
            'petro_brent_zscore',
            's&p500_eua_pct_change_lag_1m',
            'shanghai_china_pct_change_lag_1m',
        ])
        & set(features_validas)
    ),
    # 4. Apenas Indicadores de Preço do Ibovespa
    'Apenas_Ibovespa': list(
        set([coluna_base, 'ibovespa_br_momentum', 'ibovespa_br_zscore'])
        & set(features_validas)
    ),

    'Feature_Selecionados': list(
        set([coluna_base, 'ibovespa_br_momentum', 'ibovespa_br_zscore'])
        & set(features_validas)
    ),
}

# Remover grupos sem colunas
conjuntos_features = {
    nome: cols for nome, cols in conjuntos_features.items() if len(cols) > 0
}


def calcular_drawdown_durations(serie_patrimonio):
  picos = serie_patrimonio.cummax()
  em_drawdown = serie_patrimonio < picos

  duracoes = []
  duracao_atual = 0

  for flag in em_drawdown:
    if flag:
      duracao_atual += 1
    else:
      if duracao_atual > 0:
        duracoes.append(duracao_atual)
      duracao_atual = 0

  if duracao_atual > 0:
    duracoes.append(duracao_atual)

  if len(duracoes) == 0:
    return 0.0, 0.0

  return np.max(duracoes), np.mean(duracoes)


metricas_stress_com_features = []

# 7. Loop de Execução
for limite in [0.08, 0.10]:
  for regime in ['fixa', 'adaptativa']:
    for slippage in [0.003, 0.005]:
      for nome_grupo, colunas_grupo in conjuntos_features.items():

        cols_validas = [c for c in colunas_grupo if c in df_features.columns]
        if not cols_validas:
          continue

        print(
            f"Rodando: Grupo={nome_grupo} ({len(cols_validas)} cols) |"
            f" Reg={regime} | Lim={limite*100:.0f}% |"
            f" Slip={slippage*100:.1f}%..."
        )

        # ATENÇÃO: Passamos df_features completo e filtramos apenas colunas_hmm=cols_validas
        df_res, _, _, _, _ = run_walk_forward_motor(
            df_features=df_features,  # Mantém o df_features completo para o motor
            retornos_sinal=retornos_sinal,
            retornos_execucao=retornos_execucao,
            acoes_disponiveis=ativos_risco,
            colunas_hmm=cols_validas,  # O HMM TREINA APENAS COM AS COLUNAS DO GRUPO!
            colunas_operacao=colunas_operacao,
            ano_inicio_operacao=2010,
            janela_teste=1,
            metrica_otimizacao='adaptativo',
            tempo_regime=22,
            limite_max_por_ativo=limite,
            custo_corretagem=0.005,
            capital_inicial=100000,
            anos_memoria_treino=4,
            custo_slippage=slippage,
            matriz_transicao=regime,
        )

        df_res.index = pd.to_datetime(df_res.index)
        df_res = df_res.sort_index()

        # Métricas de Retorno e Risco
        patrimonio_inicial = df_res['Patrimonio'].iloc[0]
        patrimonio_final = df_res['Patrimonio'].iloc[-1]
        anos = len(df_res) / 252.0
        cagr = (patrimonio_final / patrimonio_inicial) ** (1 / anos) - 1

        retornos = df_res['Retorno_Modelo'].dropna()
        vol = retornos.std() * np.sqrt(252)
        retorno_anual_comp = (1 + retornos.mean()) ** 252 - 1
        sharpe = retorno_anual_comp / vol if vol != 0 else 0

        rolling_max = df_res['Patrimonio'].cummax()
        drawdown = (df_res['Patrimonio'] - rolling_max) / rolling_max
        max_dd = drawdown.min()

        max_duration_dias, mean_duration_dias = calcular_drawdown_durations(
            df_res['Patrimonio']
        )
        max_duration_meses = max_duration_dias / 21.0
        mean_duration_meses = mean_duration_dias / 21.0

        metricas_stress_com_features.append({
            'Grupo Features': nome_grupo,
            'N_Features': len(cols_validas),
            'Slippage (%)': f'{slippage * 100:.1f}%',
            'Regime': regime,
            'Limite (%)': f'{limite * 100:.1f}%',
            'CAGR': f'{cagr * 100:.2f}%',
            'Sharpe': f'{sharpe:.2f}',
            'Max DD': f'{max_dd * 100:.2f}%',
            'Max Duration (M)': f'{max_duration_meses:.1f} M',
            'Duração Média (M)': f'{mean_duration_meses:.1f} M',
        })

# 8. Exibir e Salvar o Relatório
df_relatorio_final = pd.DataFrame(metricas_stress_com_features)
print('\n' + '=' * 33 + ' RELATÓRIO FINAL DE ESTRESSE & FEATURES ' + '=' * 33)
print(df_relatorio_final.to_string(index=False))
print('=' * 110)

df_relatorio_final.to_csv('df_relatorio_final_simulacoes.csv')

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 1. Definir o arquivo
arquivo = BASE_DIR / 'data/gold/macro_features_hmm.parquet'
print(f'Arquivo Gold: {arquivo}')

# 2. Ler as colunas reais no Parquet
colunas_disponiveis = pd.read_parquet(arquivo).columns.tolist()

# 3. Features desejadas
features_desejadas = [
    'ibovespa_br_returns',
    'ibovespa_br_momentum',
    'ibovespa_br_zscore',
    'vix_zscore',
    'petro_brent_zscore',
    'taxa_selic_zscore',
    'risco_brasil_zscore',
    'shanghai_china_returns',
    's&p500_eua_returns',
    'inflacao_mensal_pct_change_lag_1m',
    'dolar_cambio_livre_p_tax_zscore',
    'euro_cambio_livre_zscore',
    'iene_cambio_livre_zscore',
]

features_validas = [f for f in features_desejadas if f in colunas_disponiveis]

df_features = build_feature_store(
    arquivo,
    features_validas,
    start_date='2002-08-31',
    end_date='2026-03-01',
    name_feature_store='features_model',
)

# 4. Mapear os grupos de teste
coluna_base = 'ibovespa_br_returns'

conjuntos_features = {
    'Todas_Features': features_validas,
    'Fatores_Brasil': list(
        set([
            coluna_base,
            'taxa_selic_zscore',
            'risco_brasil_zscore',
            'inflacao_mensal_pct_change_lag_1m',
            'dolar_cambio_livre_p_tax_zscore',
        ])
        & set(features_validas)
    ),
    'Fatores_Globais': list(
        set([
            coluna_base,
            'vix_zscore',
            'petro_brent_zscore',
            's&p500_eua_returns',
            'shanghai_china_returns',
            'euro_cambio_livre_zscore'
            'iene_cambio_livre_zscore',
            'dolar_cambio_livre_p_tax_zscore'
        ])
        & set(features_validas)
    ),
    'Apenas_Ibovespa': list(
        set([coluna_base, 'ibovespa_br_momentum', 'ibovespa_br_zscore'])
        & set(features_validas)
    ),
}

conjuntos_features = {
    nome: cols for nome, cols in conjuntos_features.items() if len(cols) > 0
}


def calcular_drawdown_durations(serie_patrimonio):
  picos = serie_patrimonio.cummax()
  em_drawdown = serie_patrimonio < picos
  duracoes = []
  duracao_atual = 0

  for flag in em_drawdown:
    if flag:
      duracao_atual += 1
    else:
      if duracao_atual > 0:
        duracoes.append(duracao_atual)
      duracao_atual = 0

  if duracao_atual > 0:
    duracoes.append(duracao_atual)

  if len(duracoes) == 0:
    return 0.0, 0.0

  return np.max(duracoes), np.mean(duracoes)


# Dicionário para armazenar as curvas diárias de cada combinação
resultados_por_estrategia = {}
metricas_stress_com_features = []

# 5. Loop de Execução
for limite in [0.10]:  # Focado no limite de 10%
  for regime in ['adaptativa', 'fixa']:
    for slippage in [0.005]:  # Focado em 0.5% slippage
      for nome_grupo, colunas_grupo in conjuntos_features.items():

        cols_validas = [c for c in colunas_grupo if c in df_features.columns]
        if not cols_validas:
          continue

        print(
            f'Rodando: {nome_grupo} | Regime={regime} | Limite={limite*100:.0f}%'
            '...'
        )

        df_res, _, _, _, _ = run_walk_forward_motor(
            df_features=df_features,
            retornos_sinal=retornos_sinal,
            retornos_execucao=retornos_execucao,
            acoes_disponiveis=ativos_risco,
            colunas_hmm=cols_validas,
            colunas_operacao=colunas_operacao,
            ano_inicio_operacao=2010,
            janela_teste=1,
            metrica_otimizacao='adaptativo',
            tempo_regime=22,
            limite_max_por_ativo=limite,
            custo_corretagem=0.005,
            capital_inicial=100000,
            anos_memoria_treino=4,
            custo_slippage=slippage,
            matriz_transicao=regime,
        )

        df_res.index = pd.to_datetime(df_res.index)
        df_res = df_res.sort_index()

        # Armazena o dataframe de resultado para o gráfico comparativo
        nome_curva = f'{nome_grupo} ({regime})'
        resultados_por_estrategia[nome_curva] = df_res.copy()

        # Métricas
        patrimonio_inicial = df_res['Patrimonio'].iloc[0]
        patrimonio_final = df_res['Patrimonio'].iloc[-1]
        anos = len(df_res) / 252.0
        cagr = (patrimonio_final / patrimonio_inicial) ** (1 / anos) - 1

        retornos = df_res['Retorno_Modelo'].dropna()
        vol = retornos.std() * np.sqrt(252)
        retorno_anual_comp = (1 + retornos.mean()) ** 252 - 1
        sharpe = retorno_anual_comp / vol if vol != 0 else 0

        rolling_max = df_res['Patrimonio'].cummax()
        drawdown = (df_res['Patrimonio'] - rolling_max) / rolling_max
        max_dd = drawdown.min()

        max_duration_dias, mean_duration_dias = calcular_drawdown_durations(
            df_res['Patrimonio']
        )
        max_duration_meses = max_duration_dias / 21.0
        mean_duration_meses = mean_duration_dias / 21.0

        metricas_stress_com_features.append({
            'Estratégia / Grupo': nome_grupo,
            'Regime': regime,
            'CAGR': f'{cagr * 100:.2f}%',
            'Sharpe': f'{sharpe:.2f}',
            'Max DD': f'{max_dd * 100:.2f}%',
            'Max Duration (M)': f'{max_duration_meses:.1f} M',
        })

df_relatorio_final = pd.DataFrame(metricas_stress_com_features)
print('\n' + '=' * 33 + ' RELATÓRIO FINAL ' + '=' * 33)
print(df_relatorio_final.to_string(index=False))

def plot_comparativo_estrategias(
    dict_resultados, ano_inicio=2015, tema_escuro=True
):
  """Gera um gráfico comparativo de Retorno Acumulado contínuo (%) de todas as

  estrategias/grupos testados contra o Benchmark.
  """
  fig = go.Figure()

  # Cores elegantes para diferenciar cada grupo no gráfico
  paleta_cores = [
      "#38bdf8",
      "#10b981",
      "#a855f7",
      "#f59e0b",
      "#ec4899",
      "#06b6d4",
  ]
  cor_bg = "#0b1329" if tema_escuro else "white"
  cor_texto = "#e2e8f0" if tema_escuro else "#1e293b"
  cor_grid = (
      "rgba(255, 255, 255, 0.08)" if tema_escuro else "rgba(220, 220, 220, 0.6)"
  )

  benchmark_adicionado = False

  for i, (nome_est, df_res) in enumerate(dict_resultados.items()):
    df = df_res.copy()

    if ano_inicio is not None:
      df = df[df.index.year >= ano_inicio]

    if df.empty:
      continue

    # 1. Adicionar o Benchmark Dinâmico apenas uma vez no fundo
    if not benchmark_adicionado and "Retorno_Benchmark_Hibrido_Dinamico" in df:
      df["Acumulado_Bench"] = (
          1 + df["Retorno_Benchmark_Hibrido_Dinamico"]
      ).cumprod() - 1
      df["Acumulado_Bench"] *= 100

      fig.add_trace(
          go.Scatter(
              x=df.index,
              y=df["Acumulado_Bench"],
              mode="lines",
              name="Benchmark Dinâmico",
              line=dict(color="#f97316", width=2, dash="dash"),
          )
      )
      benchmark_adicionado = True

    # 2. Calcular Retorno Acumulado (%) da Estratégia Atual
    df["Acumulado_Estrategia"] = (1 + df["Retorno_Modelo"]).cumprod() - 1
    df["Acumulado_Estrategia"] *= 100

    cor = paleta_cores[i % len(paleta_cores)]

    # Adicionar a linha da estratégia
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Acumulado_Estrategia"],
            mode="lines",
            name=nome_est,
            line=dict(color=cor, width=2.5),
        )
    )

    # Anotação final na ponta da linha
    retorno_final = df["Acumulado_Estrategia"].iloc[-1]
    fig.add_annotation(
        x=df.index[-1],
        y=retorno_final,
        text=f"{nome_est}: {retorno_final:.1f}%",
        showarrow=True,
        arrowhead=1,
        ax=40,
        ay=-10 * (i + 1),
        font=dict(color=cor, size=11, family="Arial"),
    )

  # Configuração de Layout para Apresentação
  fig.update_layout(
      template="plotly_dark" if tema_escuro else "plotly_white",
      paper_bgcolor=cor_bg,
      plot_bgcolor=cor_bg,
      font=dict(family="Arial, sans-serif", size=12, color=cor_texto),
      title=dict(
          text=(
              "<b>Comparativo de Retorno Acumulado por Grupo de Features"
              " (K-NESIAN)</b><br><sup>Desempenho histórico Out-of-Sample"
              " acumulado (%)</sup>"
          ),
          font=dict(size=16, color="#38bdf8"),
          x=0.02,
      ),
      xaxis_title="",
      yaxis_title="Retorno Acumulado (%)",
      hovermode="x unified",
      xaxis=dict(showgrid=True, gridcolor=cor_grid, zeroline=False),
      yaxis=dict(showgrid=True, gridcolor=cor_grid, zeroline=True),
      legend=dict(
          orientation="h",
          yanchor="bottom",
          y=1.02,
          xanchor="right",
          x=1,
          bgcolor="rgba(15, 23, 42, 0.8)",
      ),
      margin=dict(l=40, r=40, t=80, b=40),
  )

  fig.show()


# Chamar a função comparativa ao final das rodadas:
plot_comparativo_estrategias(resultados_por_estrategia, ano_inicio=2015)


# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('df_resultados.csv')
r_modelo = df['Retorno_Modelo']
r_hib = df['Retorno_Benchmark_Hibrido_Dinamico']

# Regressão OLS de Excesso de Retorno
X = sm.add_constant(r_hib)
ols = sm.OLS(r_modelo, X).fit()

alpha_anual = ((1 + ols.params['const']) ** 252) - 1
t_stat = ols.tvalues['const']
p_val = ols.pvalues['const']
ir = (
    (r_modelo - r_hib).mean() / (r_modelo - r_hib).std()
) * np.sqrt(252)

print(f'Alpha Anualizado: {alpha_anual:.2%}')
print(f't-statistic: {t_stat:.2f}')
print(f'p-value: {p_val:.6e} (Significativo: {p_val < 0.05})')
print(f'Information Ratio: {ir:.2f}')

# %%

print(df_resultado[['Retorno_Modelo','Retorno_Benchmark','Retorno_Benchmark_Hibrido_Dinamico']].head(10))
# %%
def calcular_decomposicao_alpha(df_resultado):
    """
    Decompõe o retorno diário do modelo em:
    1. Efeito Market Timing (Exposição a Risco vs CDI)
    2. Efeito Stock Picking (Seleção de Ações vs Ibov)
    3. Efeito Fricção (Custos de Execução e D+2)
    """
    df = df_resultado.copy()
    
    # Exposição do modelo e do benchmark (ex: bench 100% CDI -> w_acoes_bench = 0.0)
    w_acoes_mod = df['Capital_Acoes'] / df['Patrimonio']
    w_acoes_bench = 0.0  # ou 0.5 se benchmark 50/50
    
    r_ibov = df['Retorno_Benchmark']
    r_cdi = df['Retorno_Taxa_CDI'] if 'Retorno_Taxa_CDI' in df.columns else df['CDI']
    
    # Retorno médio ponderado da cesta de ações isolada
    # (Evita divisão por zero quando o modelo está 100% CDI)
    r_cesta_acoes = np.where(w_acoes_mod > 0, (df['Retorno_Modelo'] - (1 - w_acoes_mod) * r_cdi) / np.maximum(w_acoes_mod, 1e-6), 0.0)
    
    # 1. Market Timing
    df['Efeito_Timing'] = (w_acoes_mod - w_acoes_bench) * (r_ibov - r_cdi)
    
    # 2. Stock Picking
    df['Efeito_Picking'] = w_acoes_mod * (r_cesta_acoes - r_ibov)
    
    # 3. Retornos Acumulados dos Fatores
    timing_total = df['Efeito_Timing'].sum()
    picking_total = df['Efeito_Picking'].sum()
    
    return pd.DataFrame({
        'Fator': ['Market Timing (Macro/HMM)', 'Stock Picking (MDP/Seleção)', 'Retorno Total Acumulado'],
        'Contribuição Linear Estimada': [f"{timing_total:.2%}", f"{picking_total:.2%}", f"{(df['Retorno_Modelo'].sum()):.2%}"]
    })
# %%
calcular_decomposicao_alpha(df_resultado=df_resultado)
# %%

import pandas as pd
import numpy as np

def gerar_relatorio_decomposicao(df_resultado):
    df = df_resultado.copy()    
    # Retornos Totais
    r_mod = (1 + df['Retorno_Modelo']).prod() - 1
    r_ibov = (1 + df['Retorno_Benchmark']).prod() - 1
    r_din = (1 + df['Retorno_Benchmark_Hibrido_Dinamico']).prod() - 1
    
    # Extração CDI
    w = df['Exposicao_Acoes'].values
    r_cdi_dia = np.where(w < 0.99, (df['Retorno_Benchmark_Hibrido_Dinamico'] - w * df['Retorno_Benchmark']) / (1 - w + 1e-6), 0.0004)
    r_cdi = (1 + r_cdi_dia).prod() - 1

    tabela = pd.DataFrame({
        'Dimensão / Fator': [
            'Retorno Bruto Total',
            'Custo de Oportunidade (CDI)',
            'Alpha Total Gerado',
            '-> Parcela de Market Timing (HMM)',
            '-> Parcela de Stock Picking (MDP)'
        ],
        'Rentabilidade Acumulada': [
            f"{r_mod:.2%}", f"{r_cdi:.2%}", f"{(r_mod - r_cdi):.2%}",
            f"{(r_din - r_cdi):.2%}", f"{(r_mod - r_din):.2%}"
        ],
        'Participação no Alpha': [
            '-', '-', '100.0%',
            f"{((r_din - r_cdi) / (r_mod - r_cdi)):.1%}",
            f"{((r_mod - r_din) / (r_mod - r_cdi)):.1%}"
        ]
    })
    return tabela

print(gerar_relatorio_decomposicao(df_resultado))
# %%
