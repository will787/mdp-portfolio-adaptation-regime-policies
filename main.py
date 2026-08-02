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
        'ibovespa_br_returns', 'ibovespa_br_volatily', 'ibovespa_br_vol_regime',
        'vix_zscore', 'petro_brent_zscore', 'taxa_selic_zscore', 'risco_brasil_zscore',
        'shanghai_china_vol_regime', 's&p500_eua_vol_regime', 'inflacao_mensal_pct_change_lag_1m',
        'dolar_cambio_livre_p_tax_zscore', 'iene_cambio_livre_zscore'
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

# %%

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
    anos_memoria_treino=4, 
    tempo_regime=63, 
    limite_max_por_ativo=0.08,
    custo_corretagem=0.005,             
    custo_slippage=0.003,              
    capital_inicial=100000
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
df_recompensas.to_csv('df_recompensas.csv')
df_carteiras.to_csv('df_carteiras.csv')
df_pesos.to_csv('df_pesos.csv')

# %%
for regime, grupo in df_resultado.groupby("Nome_Regime"):

    ret_modelo = (1 + grupo["Retorno_Modelo"]).prod() - 1
    ret_ibov = (1 + grupo["Retorno_Benchmark"]).prod() - 1

    alpha = ret_modelo - ret_ibov

    print(f"{regime}: Alpha = {alpha * 100:.2f}%")

# %%

import pandas as pd
import numpy as np

def calcular_drawdown_durations(serie_patrimonio):
    # Encontra o pico acumulado (máximas históricas)
    picos = serie_patrimonio.cummax()
    
    # Cria uma máscara booleana: True se estiver em drawdown, False se bateu nova máxima
    em_drawdown = serie_patrimonio < picos
    
    # Calcula a duração de cada período de drawdown em dias úteis
    duracoes = []
    duracao_atual = 0
    
    for flag in em_drawdown:
        if flag:
            duracao_atual += 1
        else:
            if duracao_atual > 0:
                duracoes.append(duracao_atual)
            duracao_atual = 0
            
    # Se o backtest terminou enquanto o fundo ainda estava em drawdown:
    if duracao_atual > 0:
        duracoes.append(duracao_atual)
        
    if len(duracoes) == 0:
        return 0.0, 0.0
        
    return np.max(duracoes), np.mean(duracoes)

# Lista para consolidar os dicionários de métricas
metricas_stress = []

metricas_stress_com_duration = []

for limite in [0.08, 0.09, 0.10]:
    print(f"Processando Limite: {limite*100:.1f}%...")
    for memoria in [3,4]:
        print(f"Processando Durações: Memória = {memoria} | Limite = {limite*100:.1f}%...")
        for slippage in [0.003, 0.005]:
            print(f"Processando Durações: Memória = {memoria} | Slippage = {slippage*100:.1f}%...")
            
            # 1. Roda o seu motor quantitativo
            df_res, _, _, _, _ = run_walk_forward_motor(
                df_features=df_features, 
                retornos_sinal=retornos_sinal,
                retornos_execucao=retornos_execucao,
                acoes_disponiveis=ativos_risco,
                colunas_hmm=df_features.columns,
                colunas_operacao=colunas_operacao,
                ano_inicio_operacao=2010,
                janela_teste=1, 
                metrica_otimizacao='adaptativo',
                tempo_regime=22, 
                limite_max_por_ativo=limite,
                custo_corretagem=0.005,          
                capital_inicial=100000,
                anos_memoria_treino=memoria,
                custo_slippage=slippage
            )
            
            df_res.index = pd.to_datetime(df_res.index)
            df_res = df_res.sort_index()
            
            # 2. Métricas de retorno tradicionais
            patrimonio_inicial = df_res['Patrimonio'].iloc[0]
            patrimonio_final = df_res['Patrimonio'].iloc[-1]
            anos = len(df_res) / 252.0
            cagr = (patrimonio_final / patrimonio_inicial) ** (1 / anos) - 1
            
            retornos = df_res['Retorno_Modelo'].dropna()
            vol = retornos.std() * np.sqrt(252)
            retorno_anual_comp = (1 + retornos.mean()) ** 252 - 1
            sharpe = retorno_anual_comp / vol if vol != 0 else 0
            
            # 3. Métricas de profundidade do risco
            rolling_max = df_res['Patrimonio'].cummax()
            drawdown = (df_res['Patrimonio'] - rolling_max) / rolling_max
            max_dd = drawdown.min()
            
            # 4. NOVAS MÉTRICAS: Métricas de Tempo do Risco (Duration)
            max_duration_dias, mean_duration_dias = calcular_drawdown_durations(df_res['Patrimonio'])
            
            # Convertendo dias úteis de mercado para uma aproximação comercial em meses (21 dias úteis/mês)
            max_duration_meses = max_duration_dias / 21.0
            mean_duration_meses = mean_duration_dias / 21.0
            
            metricas_stress_com_duration.append({
                "Memória (Anos)": memoria,
                "Slippage (%)": f"{slippage * 100:.1f}%",
                "Limite (%)": f"{limite * 100:.1f}%",
                "CAGR": f"{cagr * 100:.2f}%",
                "Sharpe": f"{sharpe:.2f}",
                "Max DD": f"{max_dd * 100:.2f}%",
                "Max Duration (Meses)": f"{max_duration_meses:.1f} M",
                "Duração Média (Meses)": f"{mean_duration_meses:.1f} M"
            })

# Exibe o relatório institucional de encerramento do projeto
df_relatorio_final = pd.DataFrame(metricas_stress_com_duration)
print("\n" + "="*33 + " RELATÓRIO FINAL DE ESTRESSE & DURABILIDADE " + "="*33)
print(df_relatorio_final.to_string(index=False))
print("="*110)
# %%

df_relatorio_final.to_csv('df_relatorio_final_simulacoes.csv')


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
