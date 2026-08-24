from src.engine.portfolio import Portfolio
from src.engine.brain import MarketBrain
from src.engine.simulation import simular_janela_teste
from src.engine.metrics import cagr, sharpe, sortino, calmar, max_drawdown, vol
import pandas as pd
import numpy as np

def formatar_composicao(nome_carteira, carteiras_dict):
    """"
        Cria a composição da carteira, com seus pesos respectivos. 
        Ex: PETR4.SA 20% | BBAS3.SA 15% | PRIO3.SA 5% | CDI 60%
        Retorna uma string formatada.
    """
    pesos = carteiras_dict.get(nome_carteira, {})
    comp = []

    for ativo, peso in pesos.items():
        if peso > 0.001:
            comp.append(f"{ativo}: {peso:.2%}")

    return ' | '.join(comp)

def matriz_slippage_dinamico(retornos_sinal: pd.DataFrame, janela_vol: int = 21,  
                             slippage_base:float = 0.0010, multiplicador_estresse: float = 2.0, 
                             slippage_maximo: float = 0.05) -> pd.DataFrame:
    """"
        Resolve a dor do otimismo de execução em crises e ativos mortos. 
        Gera uma matriz temporal onde o custo de slippage acompanha a volatilidade real de cada ativo.
    """

    vol_historica = retornos_sinal.rolling(janela_vol).std()
    matriz_slippage = slippage_base + (multiplicador_estresse * vol_historica ** 2)
    matriz_slippage = matriz_slippage.clip(upper=slippage_maximo)

    # tratamento ativos mortos -> se o ativo sumiu/Nan o custo de liquidação é o teto de 5%
    matriz_slippage = matriz_slippage.fillna(slippage_maximo)

    if "CDI" not in matriz_slippage.columns:
        matriz_slippage["CDI"] = 0.0
    return matriz_slippage




def run_walk_forward_motor(df_features, retornos_sinal, retornos_execucao, acoes_disponiveis, colunas_hmm, colunas_operacao, 
                           ano_inicio_operacao=2005, janela_teste=1, metrica_otimizacao='adaptativo', 
                           anos_memoria_treino=10, tempo_regime=21, limite_max_por_ativo=0.08, limite_min_por_ativo=0.03, numero_ativos=20,
                           custo_corretagem=0.0005, custo_slippage=0.001, capital_inicial=100000, aliquota_cdi=0.225,
                           alpha_ema=0.20, margem_troca=0.20, matriz_transicao="adaptativa"):

    historico = {"backtest": [], "rodadas": [], "recompensas": [], "carteiras": [], "pesos": []}
    portfolio_global = Portfolio(capital_inicial, colunas_operacao)
    carteira_pendente_acumulada=None
    carteira_acumulada = "100_CDI"
    ano_fim_dados = int(df_features.index.year.max() -1) #tiramos -1 pra nao pegar dados de 2026 

    matriz_slippage_real = matriz_slippage_dinamico(retornos_sinal, janela_vol=tempo_regime,  
                                                    slippage_base=custo_slippage, multiplicador_estresse=2.0, slippage_maximo=0.05)


    for ano_teste_inicio in range(ano_inicio_operacao, ano_fim_dados + 1, janela_teste):
        ano_teste_fim = ano_teste_inicio + (janela_teste - 1)

        if ano_teste_fim > ano_fim_dados:
            break

        mascara_treino = (df_features.index.year >= (ano_teste_inicio - anos_memoria_treino)) & (df_features.index.year < ano_teste_inicio)
        mascara_teste = (df_features.index.year >= ano_teste_inicio) & (df_features.index.year <= ano_teste_fim)

        dados_treino_hmm = df_features[mascara_treino]
        dados_teste_hmm = df_features[mascara_teste]
        if len (dados_teste_hmm) < 2: 
            continue

        brain = MarketBrain()
        brain.treinar_e_otimizar(dados_treino_hmm, retornos_sinal, colunas_hmm, colunas_operacao,
                                tempo_regime, metrica_otimizacao, limite_max_por_ativo, limite_min_por_ativo,
                                numero_ativos, matriz_transicao=matriz_transicao)


        df_teste_acoes = retornos_execucao.loc[dados_teste_hmm.index].copy()
        df_teste_acoes = df_teste_acoes.join(df_features['ibovespa_br_returns'].rename('Retorno_Ibov'), how='inner')

        slippage_janela_teste = matriz_slippage_real.loc[dados_teste_hmm.index]

        portfolio_global, logs_backtest, logs_carteiras, cart_atual_ac, cart_pend_ac = simular_janela_teste(
            portfolio=portfolio_global, 
            df_teste_acoes=df_teste_acoes, 
            df_features_teste=dados_teste_hmm,
            brain=brain, 
            colunas_hmm=colunas_hmm, 
            colunas_operacao=colunas_operacao, 
            tempo_regime=tempo_regime, 
            custo_corretagem=custo_corretagem, 
            custo_slippage=slippage_janela_teste,
            carteira_inicial=carteira_acumulada,
            carteira_pendente_inicial=carteira_pendente_acumulada,
            margem_troca=margem_troca,
            alpha_ema=alpha_ema,
            limite_min_por_ativo=limite_min_por_ativo,
            aliquota_cdi=aliquota_cdi,
            anos_memoria=anos_memoria_treino
        )

        carteira_acumulada = cart_atual_ac
        carteira_pendente_acumulada = cart_pend_ac

        retornos_modelo = np.array([d["Retorno_Modelo"] for d in logs_backtest])
        retornos_bench = np.array([d["Retorno_Benchmark"] for d in logs_backtest])
        retornos_bench_hibrido = np.array([d['Retorno_Benchmark_Hibrido'] for d in logs_backtest])
        retorno_bench_dinamico = np.array([d['Retorno_Benchmark_Hibrido_Dinamico'] for d in logs_backtest])

        rf_medio_diario = df_teste_acoes["CDI"].mean()

        exposicoes_diarias = [d["Capital_Acoes"] / d["Patrimonio"] for d in logs_backtest if d["Patrimonio"] > 0]
        exposicao_media_acoes = np.mean(exposicoes_diarias) if len(exposicoes_diarias) > 0 else 0.0

        cagr_m = cagr(retornos_modelo)
        sharpe_m = sharpe(retornos_modelo, rf_medio_diario)
        sortino_m = sortino(retornos_modelo, rf_medio_diario)
        calmar_m = calmar(retornos_modelo)
        dd_m = max_drawdown(retornos_modelo)
        vol_m = vol(retornos_modelo)

        comp_0 = formatar_composicao(brain.politica_otima.get(0, "100_CDI"), brain.carteiras)
        comp_1 = formatar_composicao(brain.politica_otima.get(1, "100_CDI"), brain.carteiras)
        comp_2 = formatar_composicao(brain.politica_otima.get(2, "100_CDI"), brain.carteiras)
        comp_3 = formatar_composicao(brain.politica_otima.get(3, "100_CDI"), brain.carteiras)

        historico["rodadas"].append({
            'Rodada_Teste': f"{ano_teste_inicio}-{ano_teste_fim}",
            'Anos_Treino_Inclusos': int(dados_treino_hmm.index.year.nunique()),
            'Politica_Regime_0': brain.politica_otima.get(0, "100_CDI"),
            'Politica_Regime_1': brain.politica_otima.get(1, "100_CDI"),
            'Politica_Regime_2': brain.politica_otima.get(2, "100_CDI"),
            'Politica_Regime_3': brain.politica_otima.get(3, "100_CDI"),
            'Composicao_Est_0': comp_0, 'Composicao_Est_1': comp_1, 'Composicao_Est_2': comp_2, 'Composicao_Est_3': comp_3,
            'Exposicao_Media_Acoes': round(exposicao_media_acoes, 4),
            'Carrego_Medio_CDI': round(1 - exposicao_media_acoes, 4),
            'Retorno_Teste_Modelo': round(np.prod(1 + retornos_modelo) - 1, 2),
            'Retorno_Teste_Ibov': round(np.prod(1 + retornos_bench) - 1, 2),
            'Retorno_Teste_Bench_Hibrido': round(np.prod(1 + retornos_bench_hibrido) -1, 2),
            "Retorno_Teste_Bench_Dinamico": round(np.prod(1 + retorno_bench_dinamico) - 1, 2),
            'Alpha_Rodada': round((np.prod(1 + retornos_modelo) - 1) - (np.prod(1 + retorno_bench_dinamico) - 1), 2),
            'CAGR': round(cagr_m, 2), 'Volatilidade': round(vol_m, 2), 'Sharpe': round(sharpe_m, 2),
            'Sortino': round(sortino_m, 2), 'Max_Drawdown': round(dd_m, 2), 'Calmar': round(calmar_m, 2),
            "Mapa_Risco": brain.mapa_risco
        })


        # Aglutina os vetores de logs diários planos nas listas mestras
        historico["backtest"].extend(logs_backtest)
        historico["carteiras"].extend(logs_carteiras)
        historico["pesos"].extend([{"Data": d["Data"], **{k: v for k, v in d.items() if k in colunas_operacao}} for d in logs_backtest])
    # ==========================================================
    # CONVERSÃO DOS 5 DATAINFRAMES
    # ==========================================================
    df_resultado = pd.DataFrame(historico["backtest"]).set_index('Data') if historico["backtest"] else pd.DataFrame()
    df_rodadas = pd.DataFrame(historico["rodadas"]) if historico["rodadas"] else pd.DataFrame()
    df_recompensas = pd.DataFrame(historico["recompensas"]) if historico["recompensas"] else pd.DataFrame()
    df_carteiras = pd.DataFrame(historico["carteiras"]) if historico["carteiras"] else pd.DataFrame()
    df_pesos = pd.DataFrame(historico["pesos"]).set_index('Data') if historico["pesos"] else pd.DataFrame()

    return df_resultado, df_rodadas, df_recompensas, df_carteiras, df_pesos


        