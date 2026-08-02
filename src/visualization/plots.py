import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import skew, kurtosis


def plot_full_history(df_resultado, ano_inicio=None):
    """
    Gera um gráfico institucional de Retorno Acumulado (Juros Compostos)
    semelhante à visão contínua do Google Finance/Bloomberg.
    Permite filtrar a partir de um 'ano_inicio' específico.
    """
    df = df_resultado.copy()
    
    # Se o utilizador quiser ver a partir de uma data específica
    if ano_inicio is not None:
        df = df[df.index.year >= ano_inicio]
        
    if df.empty:
        print(f"Aviso: Não há dados a partir do ano {ano_inicio}.")
        return

    # Cálculo dos Juros Compostos (Efeito Bola de Neve contínuo)
    df['Acumulado_Modelo'] = (1 + df['Retorno_Modelo']).cumprod() - 1

    df['Acumulado_Benchmark_Hibrido'] = (1 + df['Retorno_Benchmark_Hibrido']).cumprod() - 1

    df['Acumulado_Benchmark_Dinamico'] = (1 + df['Retorno_Benchmark_Hibrido_Dinamico']).cumprod() - 1

    for col in [
        'Acumulado_Modelo',
        'Acumulado_Benchmark_Hibrido',
        'Acumulado_Benchmark_Dinamico'
    ]:
        df[col] *= 100

    # Texto para o título
    texto_periodo = f"{df.index.min().strftime('%b %Y')} até {df.index.max().strftime('%b %Y')}"

    fig = go.Figure()

    # Linha do Ibovespa (Benchmark)
    fig.add_trace(go.Scatter(
    x=df.index,
    y=df['Acumulado_Benchmark_Hibrido'],
    mode='lines',
    name='Benchmark 50/50',
    line=dict(color='dimgray', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Acumulado_Benchmark_Dinamico'],
        mode='lines',
        name='Benchmark Dinâmico',
        line=dict(
            color='darkorange',
            width=2,
            dash='dash'
        )
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Acumulado_Modelo'],
        mode='lines',
        name='K-nesian HMM',
        line=dict(color='royalblue', width=3)
    ))

    retorno_final_mod = df['Acumulado_Modelo'].iloc[-1]
    retorno_final_bench_hib = df['Acumulado_Benchmark_Hibrido'].iloc[-1]
    retorno_final_bench_hib_dinamico = df['Acumulado_Benchmark_Dinamico'].iloc[-1]
    
    fig.add_annotation(
        x=df.index[-1], y=retorno_final_mod,
        text=f"K-nesian: {retorno_final_mod:.1f}%",
        showarrow=True, arrowhead=1, ax=-80, ay=-45,
        font=dict(color="blue", size=12, weight="bold")
    )
    
    fig.add_annotation(
        x=df.index[-1], y=retorno_final_bench_hib,
        text=f"Ibov + CDI: {retorno_final_bench_hib:.1f}%",
        showarrow=True, arrowhead=1, ax=-80, ay=10,
        font=dict(color="gray", size=12)
    )

    fig.add_annotation(
        x=df.index[-1], y=retorno_final_bench_hib_dinamico,
        text=f"Benchmark Dinâmico: {retorno_final_bench_hib_dinamico:.1f}%",
        showarrow=True, arrowhead=1, ax=-40, ay=-75,
        font=dict(color="darkorange", size=12)
    )

    fig.update_layout(
        title=("<b>Evolução do Patrimônio Acumulado</b>"f"<br><sup>{texto_periodo}</sup>"),
        xaxis_title='Linha do Tempo',
        yaxis_title='Retorno Acumulado (%)',
        plot_bgcolor='white',
        hovermode='x unified', # Mostra uma linha vertical com os valores no dia
        xaxis=dict(showgrid=True, gridcolor="rgba(220,220,220,0.5)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(220,220,220,0.5)", zeroline=True, zerolinecolor="rgba(100,100,100,0.4)"),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255, 255, 255, 0.8)")
    )

    fig.show()

def plot_heatmap_alpha_mensal(df_resultado, benchmark="dinamico"):
    """
    Gera uma matriz de calor (Heatmap) que quebra o Alpha
    mês a mês para cada ano do histórico do backtest.
    """
    df = df_resultado.copy()
    
    df['Ano'] = df.index.year
    df['Mes'] = df.index.month
    
    def calcular_alpha_mensal(x):
        ret_modelo = (1 + x['Retorno_Modelo']).prod() - 1
        if benchmark == "dinamico":
            ret_bench = (1 + x['Retorno_Benchmark_Hibrido_Dinamico']).prod() - 1
        elif benchmark == "hibrido":
            ret_bench = (1 + x['Retorno_Benchmark_Hibrido']).prod() - 1
        elif benchmark == "ibovespa":
            ret_bench = (1 + x['Retorno_Benchmark']).prod() - 1
        else:
            raise ValueError(f"Benchmark desconhecido: {benchmark}")
        return (ret_modelo - ret_bench) * 100

    tabela_mensal = df.groupby(['Ano', 'Mes']).apply(calcular_alpha_mensal).reset_index(name='Alpha (%)')
    
    matriz_alpha = tabela_mensal.pivot(index='Ano', columns='Mes', values='Alpha (%)')
    
    meses_nomes = {
        1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
        7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
    }
    matriz_alpha = matriz_alpha.rename(columns=meses_nomes)
    
    colunas_ordenadas = [meses_nomes[i] for i in range(1, 13) if meses_nomes[i] in matriz_alpha.columns]
    matriz_alpha = matriz_alpha[colunas_ordenadas]
    
    fig = go.Figure(data=go.Heatmap(
        z=matriz_alpha.values,
        x=matriz_alpha.columns,
        y=matriz_alpha.index.astype(str),
        # Escala Divergente: Vermelho (Negativo) -> Branco (Zero) -> Verde (Positivo)
        colorscale=[[0.0, 'rgb(215,48,39)'], [0.5, 'rgb(255,255,255)'], [1.0, 'rgb(34,139,34)']],
        zmid=0, # Define o ponto médio exato da cor branca no zero
        text=matriz_alpha.round(1).values,
        texttemplate="%{text}%",
        textfont={"size": 10},
        hovertemplate="Ano: %{y}<br>Mês: %{x}<br>Alpha: %{z:.2f}%<extra></extra>"
    ))
    
    # Ajustes finos de Layout institucional
    fig.update_layout(
        title=f'<b>Raio-X de Alpha Mensal (Estratégia vs {benchmark.upper()})</b>',
        xaxis_title='Meses do Ano',
        yaxis_title='Janelas Anuais',
        height=35 * len(matriz_alpha) + 150,
        yaxis=dict(autorange="reversed")    
    )
    
    fig.show()

def analyze_recent_block(df_resultado, janela_anos=2):
    """
    Isola os últimos anos do backtest, calcula o Alpha e gera o gráfico percentual.
    df_resultado: DataFrame com as colunas 'Retorno_Modelo' e 'Retorno_Benchmark'.
    janela_anos: Quantidade de anos a considerar para o bloco recente.
    """
    ultimo_ano = int(df_resultado.index.year.max())
    ano_inicio_ultimo_bloco = ultimo_ano - (janela_anos - 1)

    df_ultimo_bloco = df_resultado[df_resultado.index.year >= ano_inicio_ultimo_bloco].copy()

    if not df_ultimo_bloco.empty:
        df_ultimo_bloco['Capital_Knesian_Recente'] = np.exp(df_ultimo_bloco['Retorno_Modelo'].cumsum())
        df_ultimo_bloco['Capital_Benchmark_Recente'] = np.exp(df_ultimo_bloco['Retorno_Benchmark_Hibrido_Dinamico'].cumsum())

        df_ultimo_bloco['Rentabilidade_Knesian_Perc'] = (df_ultimo_bloco['Capital_Knesian_Recente'] - 1) * 100
        df_ultimo_bloco['Rentabilidade_Benchmark_Perc'] = (df_ultimo_bloco['Capital_Benchmark_Recente'] - 1) * 100

        fig_recente = go.Figure()
        fig_recente.add_trace(go.Scatter(x=df_ultimo_bloco.index, y=df_ultimo_bloco['Rentabilidade_Knesian_Perc'], mode='lines', name='K-nesian Model', line=dict(color='blue', width=2.5)))
        fig_recente.add_trace(go.Scatter(x=df_ultimo_bloco.index, y=df_ultimo_bloco['Rentabilidade_Benchmark_Perc'], mode='lines', name='Benchmark (Ibovespa)', line=dict(color='gray', width=2, dash='dash')))
        fig_recente.update_layout(title=f'K-nesian Model: Retorno Acumulado Isolado ({ano_inicio_ultimo_bloco} - {ultimo_ano})', yaxis_title='Rentabilidade Acumulada (%)', xaxis_title='Data', template='plotly_white', hovermode='x unified')
        fig_recente.show()

        rentabilidade_knesian = df_ultimo_bloco['Rentabilidade_Knesian_Perc'].iloc[-1]
        rentabilidade_bench = df_ultimo_bloco['Rentabilidade_Benchmark_Perc'].iloc[-1]

        print("\n" + "="*45)
        print(f"📊 PERFORMANCE DO ÚLTIMO BLOCO ({ano_inicio_ultimo_bloco}-{ultimo_ano})")
        print("="*45)
        print(f"Retorno Acumulado K-nesian : {rentabilidade_knesian:>7.2f}%")
        print(f"Retorno Acumulado Benchmark: {rentabilidade_bench:>7.2f}%")
        print(f"Alpha Líquido (Diferença)  : {(rentabilidade_knesian - rentabilidade_bench):>7.2f}%")
        print("="*45)
        
        print("\n--- AS CARTEIRAS FAVORITAS DO K-NESIAN ---")
        print(df_resultado['Alocacao'].value_counts().head(10))
        
        df_ultimo_bloco.to_csv('resultado_ultimo_bloco_sharpe.csv')


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def analise_comparativa_benchmark_flag(df_resultado, df_features, flag_vline=1,):
    """
        Gera um painel comparativo exibindo o período de Warm-up (Treino cego)
        e o momento exato em que o Modelo HMM começa a usar os sinais operacionais.
        """
    df = df_resultado.copy()
    
    # Datas chave
    data_inicio_teste = df.index.min()
    data_inicio_tudo = df_features.index.min()
    
    # 1. Monta a curva completa do Ibovespa (Desde 2002)
    ibov_completo = (1 + df_features['ibovespa_br_returns']).cumprod()
    fator_rebase_ibov = ibov_completo.loc[data_inicio_teste]
    ibov_base_100 = (ibov_completo / fator_rebase_ibov) * 100

    # 2. Monta a curva do Modelo (Começa apenas no Início do Teste)
    df['Capital_Modelo'] = (1 + df['Retorno_Modelo']).cumprod() * 100
    
    # 3. Métricas Anuais
    df['Ano'] = df.index.year
    def metricas_anuais(x):
        ret_modelo = (1 + x['Retorno_Modelo']).prod() - 1
        ret_bench = (1 + x['Retorno_Benchmark_Hibrido_Dinamico']).prod() - 1
        vol_modelo = x['Retorno_Modelo'].std() * np.sqrt(252)
        sharpe_modelo = (x['Retorno_Modelo'].mean() * 252) / (vol_modelo + 1e-6)
        
        return pd.Series({
            'Retorno_Modelo (%)': ret_modelo * 100,
            'Retorno_Ibov (%)': ret_bench * 100,
            'Alpha (%)': (ret_modelo - ret_bench) * 100,
            'Volatilidade (%)': vol_modelo * 100,
            'Sharpe': sharpe_modelo
        })
        
    tabela_anos = df.groupby('Ano').apply(metricas_anuais).reset_index()

    # 4. Montagem do Painel Visual
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=('Equity Curve: Aprendizado (Cego) vs Operação HMM', 'Alpha Anual (Modelo vs Benchmark Dinâmico)'),
        row_heights=[0.7, 0.3]
    )        

    # Gráfico Superior: Ibovespa Completo
    fig.add_trace(go.Scatter(x=df_features.index, y=ibov_base_100, mode='lines', name='Ibovespa (Histórico Completo)', line=dict(color='gray', width=1.5, dash='dash')), row=1, col=1)
    
    # Gráfico Superior: Modelo (K-nesian)
    fig.add_trace(go.Scatter(x=df.index, y=df['Capital_Modelo'], mode='lines', name='K-nesian Pós-Warmup (Sinais HMM)', line=dict(color='blue', width=2.5)), row=1, col=1)

    if flag_vline == 1:
        # A TÁTICA INFALÍVEL: Usamos um Scatter com apenas 2 pontos para desenhar a linha vertical.
        # Assim ela aparece perfeitamente no gráfico E na legenda.
        max_y = max(ibov_base_100.max(), df['Capital_Modelo'].max()) * 1.05 # Ponto mais alto do gráfico
        
        fig.add_trace(go.Scatter(
            x=[data_inicio_teste, data_inicio_teste], 
            y=[0, max_y], 
            mode='lines', 
            name='Fim do Warm-up / Início HMM', 
            line=dict(color='red', width=2, dash='dash')
        ), row=1, col=1)
        
        # BÔNUS: Sombreamento cinza para destacar todo o período em que o modelo estava "estagiando"
        fig.add_vrect(
            x0=data_inicio_tudo, x1=data_inicio_teste,
            fillcolor="lightgray", opacity=0.3,
            layer="below", line_width=0,
            annotation_text="Treino HMM<br>(Warm-up)", annotation_position="top left",
            row=1, col=1
        )

    # Gráfico Inferior: Barras de Alpha
    cores_alpha = ['green' if val > 0 else 'red' for val in tabela_anos['Alpha (%)']]
    fig.add_trace(go.Bar(
        x=tabela_anos['Ano'].astype(str), y=tabela_anos['Alpha (%)'], 
        marker_color=cores_alpha, name='Alpha Anual',
        text=tabela_anos['Alpha (%)'].round(1).astype(str) + '%', textposition='auto'
    ), row=2, col=1)

    # Ajustes de Layout
    fig.update_layout(title_text="Análise de Performance Quantitativa", height=800, template="plotly_white", hovermode="x unified")
    fig.update_yaxes(title_text="Capital Total (R$)", row=1, col=1)
    fig.update_yaxes(title_text="Diferença de Retorno (%)", row=2, col=1)
    
    fig.show()


def analise_comparativa_benchmark(df_resultado):
    """
    Gera um painel comparativo entre a Estratégia e o Benchmark (Ibovespa Hibrido),
    incluindo a curva de capital acumulada e a performance detalhada por janelas de teste (anos).
    """
    df = df_resultado.copy()

    # 1. Cálculo da Curva de Capital (Equity Curve) Base 100
    df['Capital_Modelo'] = (1 + df['Retorno_Modelo']).cumprod() * 100
    df['Capital_Benchmark'] = (1 + df['Retorno_Benchmark_Hibrido_Dinamico']).cumprod() * 100
    
    # 2. Cálculo de Performance por Ano (Janelas de Teste)
    df['Ano'] = df.index.year
    
    def metricas_anuais(x):
        ret_modelo = (1 + x['Retorno_Modelo']).prod() - 1
        ret_bench = (1 + x['Retorno_Benchmark_Hibrido_Dinamico']).prod() - 1
        vol_modelo = x['Retorno_Modelo'].std() * np.sqrt(252)
        sharpe_modelo = (x['Retorno_Modelo'].mean() * 252) / (vol_modelo + 1e-6)
        
        return pd.Series({
            'Retorno_Modelo (%)': ret_modelo * 100,
            'Retorno_Ibov (%)': ret_bench * 100,
            'Alpha (%)': (ret_modelo - ret_bench) * 100,
            'Volatilidade (%)': vol_modelo * 100,
            'Sharpe': sharpe_modelo
        })
        
    tabela_anos = df.groupby('Ano').apply(metricas_anuais).reset_index()
    
    # Exibe a tabela no terminal para conferência rápida
    print("\n" + "="*60)
    print(" PERFORMANCE POR JANELA DE TESTE (OUT-OF-SAMPLE) ")
    print("="*60)
    print(tabela_anos.round(2).to_string(index=False))
    print("="*60 + "\n")

    # 3. Montagem do Painel Visual
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=('Curva de Capital Acumulada (Base 100)', 'Alpha Anual (Modelo vs Ibovespa)'),
        row_heights=[0.7, 0.3]
    )        

    # Gráfico Superior: Curva de Capital
    fig.add_trace(go.Scatter(x=df.index, y=df['Capital_Modelo'], mode='lines', name='K-nesian Modelo', line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Capital_Benchmark'], mode='lines', name='Benchmark (Ibovespa)', line=dict(color='gray', width=1.5, dash='dash')), row=1, col=1)

    # Gráfico Inferior: Barras de Alpha (Verde se bateu o Ibov, Vermelho se perdeu)
    cores_alpha = ['green' if val > 0 else 'red' for val in tabela_anos['Alpha (%)']]
    fig.add_trace(go.Bar(
        x=tabela_anos['Ano'].astype(str), 
        y=tabela_anos['Alpha (%)'], 
        marker_color=cores_alpha,
        name='Alpha Anual',
        text=tabela_anos['Alpha (%)'].round(1).astype(str) + '%',
        textposition='auto'
    ), row=2, col=1)

    # Ajustes de Layout
    fig.update_layout(
        title_text="Análise de Performance: Estratégia HMM vs Mercado",
        height=800,
        template="plotly_white",
        hovermode="x unified",
        showlegend=True
    )
    
    fig.update_yaxes(title_text="Capital Total (R$)", row=1, col=1)
    fig.update_yaxes(title_text="Diferença de Retorno (%)", row=2, col=1)
    
    fig.show()

def plot_regimes_historicos(df_resultado):

    df_plot = df_resultado.copy()

    df_plot['Ibovespa_Acumulado'] = (
        1 + df_plot['Retorno_Benchmark']
    ).cumprod() * 100


    fig = px.scatter(
        df_plot,
        x=df_plot.index,
        y='Ibovespa_Acumulado',
        color='Nome_Regime',
        title='Raio-X Macro: Regimes Detectados pelo HMM',
        log_y=True
    )

    fig.show()


def plot_heatmap_alpha_mensal(df_resultado):
    """
    Gera uma matriz de calor (Heatmap) institucional que quebra o Alpha
    mês a mês para cada ano do histórico do backtest.
    """
    df = df_resultado.copy()
    
    df['Ano'] = df.index.year
    df['Mes'] = df.index.month
    
    def calcular_alpha_mensal(x):
        ret_modelo = (1 + x['Retorno_Modelo']).prod() - 1
        ret_bench = (1 + x['Retorno_Benchmark_Hibrido_Dinamico']).prod() - 1
        return (ret_modelo - ret_bench) * 100

    tabela_mensal = df.groupby(['Ano', 'Mes']).apply(calcular_alpha_mensal).reset_index(name='Alpha (%)')
    
    matriz_alpha = tabela_mensal.pivot(index='Ano', columns='Mes', values='Alpha (%)')
    
    meses_nomes = {
        1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
        7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
    }
    matriz_alpha = matriz_alpha.rename(columns=meses_nomes)
    
    colunas_ordenadas = [meses_nomes[i] for i in range(1, 13) if meses_nomes[i] in matriz_alpha.columns]
    matriz_alpha = matriz_alpha[colunas_ordenadas]
    
    fig = go.Figure(data=go.Heatmap(
        z=matriz_alpha.values,
        x=matriz_alpha.columns,
        y=matriz_alpha.index.astype(str),
        # Escala Divergente: Vermelho (Negativo) -> Branco (Zero) -> Verde (Positivo)
        colorscale=[[0.0, 'rgb(215,48,39)'], [0.5, 'rgb(255,255,255)'], [1.0, 'rgb(34,139,34)']],
        zmid=0, # Define o ponto médio exato da cor branca no zero
        text=matriz_alpha.round(1).values,
        texttemplate="%{text}%",
        textfont={"size": 10},
        hovertemplate="Ano: %{y}<br>Mês: %{x}<br>Alpha: %{z:.2f}%<extra></extra>"
    ))
    
    # Ajustes finos de Layout institucional
    fig.update_layout(
        title='<b>Raio-X de Alpha Mensal (Estratégia HMM vs Ibovespa)</b>',
        xaxis_title='Meses do Ano',
        yaxis_title='Janelas Anuais',
        height=35 * len(matriz_alpha) + 150,
        yaxis=dict(autorange="reversed")    
    )
    
    fig.show()

def plot_scatter_retornos_mensais(df_resultado, coluna_bench='Retorno_Benchmark',ano_destaque=2025, ano_inicio=None, ano_fim=None):
    """
    Gera um gráfico de dispersão (Scatter Plot) comparando o retorno mensal do Modelo
    vs o Benchmark. Adiciona uma regressão linear (OLS) e destaca um ano específico.
    Permite filtrar o período visualizado usando ano_inicio e ano_fim.
    """
    df = df_resultado.copy()
    
    # Usa 'ME' (Month End) para alinhar ao final de cada mês
    df_mensal = df[['Retorno_Modelo', coluna_bench]].resample('ME').apply(
        lambda x: ((1 + x).prod() - 1) * 100
    )
    
    df_mensal['Ano'] = df_mensal.index.year
    df_mensal['Mes'] = df_mensal.index.month
    
    if ano_inicio is not None:
        df_mensal = df_mensal[df_mensal['Ano'] >= ano_inicio]
    if ano_fim is not None:
        df_mensal = df_mensal[df_mensal['Ano'] <= ano_fim]
    
    def categorizar_ano(ano):
        if ano == ano_destaque:
            return f'Destaque ({ano_destaque})'
        return 'Histórico Base'
        
    df_mensal['Categoria'] = df_mensal['Ano'].apply(categorizar_ano)
    meses_nomes = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    df_mensal['Nome_Mes'] = df_mensal['Mes'].apply(lambda x: meses_nomes[x-1])
    periodo_texto = "Histórico Completo"

    if ano_inicio is not None and ano_fim is not None:
        periodo_texto = f"{ano_inicio} a {ano_fim}"
    elif ano_inicio is not None:
        periodo_texto = f"A partir de {ano_inicio}"
    elif ano_fim is not None:
        periodo_texto = f"Até {ano_fim}"
    
    fig = px.scatter(
        df_mensal,
        x=coluna_bench,
        y='Retorno_Modelo',
        color='Categoria',
        trendline='ols', # Adiciona a Regressão Linear Mínimos Quadrados
        trendline_scope='overall', # Calcula uma única reta de regressão para todos os dados
        hover_data=['Ano', 'Nome_Mes'],
        color_discrete_map={
            'Histórico Base': 'rgba(100, 149, 237, 0.6)', # Azul claro semi-transparente
            f'Destaque ({ano_destaque})': 'rgba(255, 69, 0, 1.0)' # Laranja vibrante
        },
        labels={
            'Retorno_Benchmark': 'Benchmark: Retorno Mensal (%)',
            'Retorno_Modelo': 'HMM K-nesian: Retorno Mensal (%)',
            'Categoria': 'Período'
        },
        title=f'<b>Dispersão de Retornos Mensais e Regressão Linear ({periodo_texto})</b><br><sup>Destaque visual para o ano de {ano_destaque}</sup>'
    )
    
    # Extrair e exibir os parâmetros Matemáticos da Regressão ---
    resultados_ols = px.get_trendline_results(fig)
    if not resultados_ols.empty:
        modelo_ols = resultados_ols.iloc[0]["px_fit_results"]
        params = modelo_ols.params
        
        # Tratamento de compatibilidade: dependendo da versão, params pode ser ndarray ou Series
        if hasattr(params, 'iloc'):
            alpha_estrutural = params.iloc[0]
            beta_mercado = params.iloc[1]
        else:
            alpha_estrutural = params[0]
            beta_mercado = params[1]
        
        # Adicionar a caixa com as métricas extraídas no gráfico
        fig.add_annotation(
            x=0.02, y=0.98, # Canto superior esquerdo
            xref="paper", yref="paper",
            text=f"<b>Alpha Estrutural:</b> {alpha_estrutural:.2f}% ao mês<br><b>Beta (Exposição):</b> {beta_mercado:.2f}",
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
            font=dict(size=12, color="black")
        )

    # 4. Ajustes finos institucionais
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='black', zerolinewidth=2),
        yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='black', zerolinewidth=2),
        hovermode='closest'
    )
    
    # Adiciona uma reta diagonal pontilhada neutra (y = x) para referência de "Empate"
    min_val = min(df_mensal[coluna_bench].min(), df_mensal['Retorno_Modelo'].min()) - 2
    max_val = max(df_mensal[coluna_bench].max(), df_mensal['Retorno_Modelo'].max()) + 2
    
    fig.add_shape(
        type='line',
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(color='gray', dash='dash'),
        opacity=0.5
    )
    fig.add_annotation(
        x=max_val - 2, y=max_val,
        text="Linha de Empate (Alpha = 0)",
        showarrow=False,
        font=dict(color="gray", size=10)
    )

    fig.show()

    # ==========================================
    # 5. PRINT ESTRUTURADO DE ESTATÍSTICAS
    # ==========================================
    ret_modelo_dec = df_mensal['Retorno_Modelo'] / 100
    ret_bench_dec = df_mensal[coluna_bench] / 100
    acum_modelo = (1 + ret_modelo_dec).prod() - 1
    acum_bench = (1 + ret_bench_dec).prod() - 1
    alpha_total = acum_modelo - acum_bench
    alpha_mensal_serie = df_mensal['Retorno_Modelo'] - df_mensal[coluna_bench]
    alpha_medio = alpha_mensal_serie.mean()
    meses_vitoriosos = (alpha_mensal_serie > 0).sum()
    total_meses = len(df_mensal)
    hit_ratio = (meses_vitoriosos / total_meses) * 100 if total_meses > 0 else 0 
    print("\n" + "="*55)
    print(f"📊 RESUMO ESTATÍSTICO DO PERÍODO: {periodo_texto}")
    print("="*55)
    print(f"Total de Meses Analisados : {total_meses} meses")
    print(f"Retorno Acumulado Modelo  : {acum_modelo*100:+.2f}%")
    print(f"Retorno Acumulado Mercado : {acum_bench*100:+.2f}%")
    print(f"Alpha Total (Acumulado)   : {alpha_total*100:+.2f}%")
    print("-" * 55)
    print(f"Alpha Médio (Aritmético)  : {alpha_medio:+.2f}% ao mês")
    print(f"Alpha Estrutural (OLS)    : {alpha_estrutural:+.2f}% ao mês")
    print(f"Beta em relação ao Mercado: {beta_mercado:.2f}")
    print(f"Hit Ratio (Vitórias)      : {hit_ratio:.1f}% ({meses_vitoriosos}/{total_meses} meses)")
    print("="*55 + "\n")

def plot_distribuicao_retornos(df_resultado, bench_hibrido=True):
    """
    Mostra o formato da distribuição (caudas gordas, assimetria) incluindo o Benchmark Híbrido.
    """
    ret_mod = df_resultado['Retorno_Modelo'] * 100
    ret_bench = df_resultado['Retorno_Benchmark_Hibrido_Dinamico'] * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=ret_bench, histnorm='probability density', name='Benchmark Dinâmico',
        marker_color='gray', opacity=0.3, nbinsx=100
    ))
    
    has_hib = bench_hibrido and 'Retorno_Benchmark_Hibrido' in df_resultado.columns
    if has_hib:
        ret_hib = df_resultado['Retorno_Benchmark_Hibrido'] * 100
        fig.add_trace(go.Histogram(
            x=ret_hib, histnorm='probability density', name='Bench Híbrido (50/50)',
            marker_color='orange', opacity=0.4, nbinsx=100
        ))
    
    fig.add_trace(go.Histogram(
        x=ret_mod, histnorm='probability density', name='K-nesian (Modelo)',
        marker_color='#1f77b4', opacity=0.6, nbinsx=100
    ))
    
    fig.update_layout(
        title='<b>Distribuição Diária de Retornos (Densidade)</b><br><sup>Avaliação do Risco de Cauda (Crashes)</sup>',
        xaxis_title='Retorno Diário (%)',
        yaxis_title='Densidade de Probabilidade',
        barmode='overlay',
        plot_bgcolor='white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='black'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey')
    )
    fig.show()
    
    print("\n" + "="*65)
    print("📊 MOMENTOS ESTATÍSTICOS DA DISTRIBUIÇÃO (DIÁRIO)")
    print("="*65)
    print(f"{'Estratégia':<25} | {'Skewness':<10} | {'Kurtosis':<10}")
    print("-" * 65)
    print(f"{'Benchmark Dinâmico':<25} | {skew(ret_bench):10.2f} | {kurtosis(ret_bench):10.2f}")
    if has_hib:
        print(f"{'Bench Híbrido (50/50)':<25} | {skew(ret_hib):10.2f} | {kurtosis(ret_hib):10.2f}")
    print(f"{'K-nesian (Modelo)':<25} | {skew(ret_mod):10.2f} | {kurtosis(ret_mod):10.2f}")
    print("="*65)
    print("*Nota Quant:* Assimetria positiva (Skew > 0) indica maior frequência de retornos positivos;\n"
          "             Curtose alta (Kurt > 3) indica caudas gordas (maior risco de crash).")


def analise_atribuicao_alpha(df_resultado):
    """
    Decompõe o retorno considerando que a exposição a ações varia de forma contínua (0 a 1).
    - Market Timing: Capturado pelo ganho/perda de se alocar em CDI vs. Ações proporcionalmente à exposição.
    - Stock Picking: Capturado pelo excesso de retorno da carteira de ações do modelo vs. Benchmark quando exposto ao risco.
    """
    df = df_resultado.copy()
    
    # Garantir que a exposição está entre 0 e 1 (caso esteja em %)
    if df['Exposicao_Acoes'].max() > 1.0:
        df['Exposicao_Acoes'] = df['Exposicao_Acoes'] / 100.0
        
    w = df['Exposicao_Acoes'] # Peso no ativo de risco (ex: 0.7 = 70% ações, 30% CDI)
    
    # Retornos diários
    ret_mod = df['Retorno_Modelo']
    ret_bench = df['Retorno_Benchmark_Hibrido_Dinamico']
    
    # Como a exposição varia, podemos estimar o retorno do benchmark puramente em ações 
    # e puramente no CDI se necessário, mas para manter a robustez com as colunas existentes:
    # Vamos isolar o impacto da variação de peso (Timing) vs Seleção (Picking) dia a dia.
    
    # Exposição média do período
    exp_media = w.mean() * 100
    
    # Acumulados globais
    acum_mod = (1 + ret_mod).prod() - 1
    acum_bench = (1 + ret_bench).prod() - 1
    
    print("\n" + "="*65)
    print("🔍 PERFORMANCE ATTRIBUTION (EXPOSIÇÃO CONTÍNUA)")
    print("="*65)
    print(f"• Exposição Média a Risco no Período: {exp_media:.1f}%")
    print(f"• Exposição Mínima registrada:       {w.min()*100:.1f}%")
    print(f"• Exposição Máxima registrada:       {w.max()*100:.1f}%")
    print("-" * 65)
    print(f"{'Estratégia / Retorno Global':<30} | {'Retorno Acumulado':<15}")
    print("-" * 65)
    print(f"{'K-nesian (Modelo)':<30} | {acum_mod*100:+15.2f}%")
    print(f"{'Benchmark Híbrido Dinâmico':<30} | {acum_bench*100:+15.2f}%")
    print(f"{'Alpha Total Gerado':<30} | {(acum_mod - acum_bench)*100:+15.2f}%")
    print("="*65)
    
    # Visão por Faixas de Exposição (para entender o comportamento tático)
    df['Faixa_Exposicao'] = pd.cut(
        w, 
        bins=[-0.01, 0.25, 0.75, 0.99, 1.01], 
        labels=['Defesa Forte (<25%)', 'Defesa Parcial (25-75%)', 'Ataque Parcial (75-99%)', 'Ataque Total (100%)']
    )
    
    print("\n📊 **Comportamento por Regime de Exposição:**")
    print(f"{'Regime':<28} | {'Tempo (%):':<10} | {'Retorno Médio Diário':<20}")
    print("-" * 65)
    
    resumo_faixas = df.groupby('Faixa_Exposicao', observed=True).agg(
        Tempo=('Retorno_Modelo', 'count'),
        Ret_Medio_Mod=('Retorno_Modelo', lambda x: (x.mean() * 100))
    )
    resumo_faixas['Pct_Tempo'] = (resumo_faixas['Tempo'] / len(df)) * 100
    for faixa, row in resumo_faixas.iterrows():
        print(f"{str(faixa):<28} | {row['Pct_Tempo']:9.1f}% | {row['Ret_Medio_Mod']:+18.4f}% ao dia")
    
    print("="*65 + "\n")

def plot_evolucao_alocacao(df_resultado):
    """
    Gráfico de área para visualizar o nível de exposição real a risco (Bolsa vs Caixa).
    Classifica o apetite ao risco com base na coluna Exposicao_Acoes.
    """
    df = df_resultado.copy()
    
    # Se a exposição estiver de 0.0 a 1.0, multiplica por 100. Se já estiver 0 a 100, mantém.
    if df['Exposicao_Acoes'].max() <= 1.5:
        df['Exposicao_Acoes'] = df['Exposicao_Acoes'] * 100
        
    # CORREÇÃO: Lógica sequencial contínua para capturar qualquer número decimal (floats)
    def classificar_modo(exposicao):
        if exposicao >= 80: 
            return 'Ataque Total (>80% Ações)'
        elif exposicao >= 60: 
            return 'Ataque Moderado (60-79% Ações)'
        elif exposicao >= 40: 
            return 'Neutro / Balanceado (40-59% Ações)'
        else: 
            return 'Proteção Máxima (<40% Ações)'
        
    df['Modo_Alocacao'] = df['Exposicao_Acoes'].apply(classificar_modo)
    
    # Contar a frequência de cada modo por mês
    df_mensal = df.groupby([pd.Grouper(freq='ME'), 'Modo_Alocacao']).size().unstack(fill_value=0)
    
    # Converter para percentual (0 a 100%) do mês
    df_mensal_pct = df_mensal.div(df_mensal.sum(axis=1), axis=0) * 100
    
    # Garantir que todas as colunas existem para a ordem de cores não quebrar
    categorias = [
        'Ataque Total (>80% Ações)',
        'Ataque Moderado (60-79% Ações)',
        'Neutro / Balanceado (40-59% Ações)',
        'Proteção Máxima (<40% Ações)'
    ]
    
    for cat in categorias:
        if cat not in df_mensal_pct.columns:
            df_mensal_pct[cat] = 0.0
            
    # Reordenar para o gráfico de área ficar lógico 
    df_mensal_pct = df_mensal_pct[categorias]
    
    fig = px.area(
        df_mensal_pct, 
        color_discrete_map={
            'Ataque Total (>80% Ações)': 'rgba(5, 12, 156, 0.8)',       # Azul Marinho Profundo
            'Ataque Moderado (60-79% Ações)': 'rgba(53, 114, 239, 0.8)', # Azul Royal
            'Neutro / Balanceado (40-59% Ações)': 'rgba(230, 126, 34, 0.8)', # Laranja
            'Proteção Máxima (<40% Ações)': 'rgba(191, 49, 49, 0.8)'     # Vermelho Defesa
        },
        labels={'value': '% dos Dias no Mês', 'variable': 'Perfil de Risco', 'Data': 'Ano'}
    )
    
    fig.update_layout(
        title="Dinâmica de Alocação (Market Timing) ao longo do tempo",
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgrey', title="Linha do Tempo"),
        yaxis=dict(showgrid=True, gridcolor='lightgrey', range=[0, 100]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    fig.show()

def calcular_retornos_anuais(df_resultado):
    """
    Agrupa os retornos diários por ano e calcula o retorno acumulado de cada ano.
    """
    df = df_resultado.copy()
    
    # Garante que o índice é datetime para extrair o ano
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
        
    df['Ano'] = df.index.year
    
    # Agrupa por ano e calcula o retorno composto: (1+r1)*(1+r2)... - 1
    df_anual = df.groupby('Ano')[['Retorno_Modelo', 'Retorno_Benchmark']].apply(
        lambda x: pd.Series({
            'Retorno_Modelo_Ano': (np.prod(1 + x['Retorno_Modelo']) - 1) * 100,
            'Retorno_Ibov_Ano': (np.prod(1 + x['Retorno_Benchmark']) - 1) * 100
        })
    ).reset_index()
    
    return df_anual

import plotly.graph_objects as go

def plot_retorno_anual_comparativo(df_resultado):
    # 1. Prepara os dados
    df_anual = calcular_retornos_anuais(df_resultado)
    
    # Filtra para ignorar anos com dados incompletos (opcional)
    # df_anual = df_anual[df_anual['Ano'] >= 2011]
    
    # 2. Cria a figura
    fig = go.Figure()
    
    # Barra do Modelo (K-nesian)
    fig.add_trace(go.Bar(
        x=df_anual['Ano'],
        y=df_anual['Retorno_Modelo_Ano'],
        name='K-nesian (Modelo)',
        marker_color='blue',
        text=df_anual['Retorno_Modelo_Ano'].apply(lambda x: f'{x:+.1f}%'),
        textposition='auto',
        textfont=dict(color='white')
    ))
    
    # Barra do Benchmark (Ibovespa)
    fig.add_trace(go.Bar(
        x=df_anual['Ano'],
        y=df_anual['Retorno_Ibov_Ano'],
        name='Ibovespa',
        marker_color='gray',
        text=df_anual['Retorno_Ibov_Ano'].apply(lambda x: f'{x:+.1f}%'),
        textposition='auto',
        textfont=dict(color='white')
    ))
    
    # 3. Formata o Layout para agrupar as barras
    fig.update_layout(
        title='Retorno Acumulado Anual: Modelo vs Ibovespa',
        barmode='group', # <--- ISSO COLOCA AS BARRAS LADO A LADO
        plot_bgcolor='white',
        xaxis=dict(
            title='Ano',
            tickmode='linear',
            tick0=df_anual['Ano'].min(),
            dtick=1,
            showgrid=False
        ),
        yaxis=dict(
            title='Retorno no Ano (%)',
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    fig.show()

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_dashboard_rodadas(df_rodadas):
    """
    Gera um dashboard quantitativo completo comparando as métricas 
    ano a ano (por rodada de teste) extraídas do df_rodadas.
    """
    df = df_rodadas.copy()
    
    # Limpa a string da rodada para mostrar apenas o ano no eixo X (ex: "2011-2011" -> "2011")
    df['Ano'] = df['Rodada_Teste'].apply(lambda x: str(x).split('-')[0])
    
    # Cria os subplots (3 linhas, 1 coluna)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "1. Retorno Anual (Modelo vs Benchmark)", 
            "2. Eficiência (Sharpe e Sortino do Modelo)", 
            "3. Risco (Volatilidade e Max Drawdown do Modelo)"
        )
    )

    # ---------------------------------------------------------
    # LINHA 1: Retorno Modelo vs Retorno Ibov (Gráfico de Barras Agrupadas)
    # ---------------------------------------------------------
    fig.add_trace(go.Bar(
        x=df['Ano'], y=df['Retorno_Teste_Modelo'] * 100,
        name='K-nesian (Modelo)',
        marker_color='blue',
        text=df['Retorno_Teste_Modelo'].apply(lambda x: f'{x*100:+.1f}%'),
        textposition='auto',
        textfont=dict(color='white', size=10)
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df['Ano'], y=df['Retorno_Teste_Ibov'] * 100,
        name='Ibovespa',
        marker_color='gray',
        text=df['Retorno_Teste_Ibov'].apply(lambda x: f'{x*100:+.1f}%'),
        textposition='auto',
        textfont=dict(color='white', size=10)
    ), row=1, col=1)


    # ---------------------------------------------------------
    # LINHA 2: Indicadores de Risco-Retorno (Gráfico de Linhas)
    # ---------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=df['Ano'], y=df['Sharpe'],
        name='Índice de Sharpe',
        mode='lines+markers+text',
        marker=dict(color='purple', size=8),
        line=dict(width=2),
        text=df['Sharpe'].apply(lambda x: f'{x:.2f}'),
        textposition='top center'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df['Ano'], y=df['Sortino'],
        name='Índice de Sortino',
        mode='lines+markers+text',
        marker=dict(color='green', size=8),
        line=dict(width=2, dash='dot'),
        text=df['Sortino'].apply(lambda x: f'{x:.2f}'),
        textposition='bottom center'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df['Ano'], y=df['Calmar'],
        name='Índice de Calmar',
        mode='lines+markers+text',
        marker=dict(color='darkorange', size=8),
        line=dict(width=2, dash='dash'),
        text=df['Calmar'].apply(lambda x: f'{x:.2f}'),
        textposition='bottom center'
    ), row=2, col=1)

    # ---------------------------------------------------------
    # LINHA 3: Métricas de Risco (Volatilidade e Drawdown)
    # ---------------------------------------------------------
    fig.add_trace(go.Bar(
        x=df['Ano'], y=df['Volatilidade'] * 100,
        name='Volatilidade Anualizada',
        marker_color='rgba(255, 165, 0, 0.6)', # Laranja translúcido
        text=df['Volatilidade'].apply(lambda x: f'{x*100:.1f}%'),
        textposition='auto'
    ), row=3, col=1)

    # Invertemos o Drawdown (multiplicando por -1) para o gráfico mostrar "o quão fundo" a carteira caiu
    fig.add_trace(go.Bar(
        x=df['Ano'], y=df['Max_Drawdown'] * -100,
        name='Max Drawdown',
        marker_color='rgba(220, 20, 60, 0.8)', # Vermelho Crimson
        text=df['Max_Drawdown'].apply(lambda x: f'-{x*100:.1f}%'),
        textposition='outside'
    ), row=3, col=1)

    # ---------------------------------------------------------
    # LAYOUT FINAL
    # ---------------------------------------------------------
    fig.update_layout(
        title_text="K-nesian Walk-Forward: Avaliação Quantitativa por Rodada",
        height=1000,  # Aumenta a altura para comportar as 3 linhas de forma confortável
        plot_bgcolor='white',
        barmode='group',
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    # Formatação dos eixos Y para exibir os sufixos corretos
    fig.update_yaxes(title_text="Retorno (%)", row=1, col=1, showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='black')
    fig.update_yaxes(title_text="Ratio", row=2, col=1, showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='black')
    fig.update_yaxes(title_text="Risco (%)", row=3, col=1, showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='black')

    # Mantém o eixo X sem gridlines para um visual mais limpo
    fig.update_xaxes(showgrid=False)

    fig.show()

