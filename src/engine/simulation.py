import numpy as np
import sys
from src.engine.strategy import executar_estrategia
from src.engine.benchmark import benchmark_hibrido, atribuicao_benchmark_dinamico
from src.engine.brain import MarketBrain
import pandas as pd


def simular_janela_teste(portfolio, df_teste_acoes, df_features_teste, brain, colunas_hmm, colunas_operacao, tempo_regime,
                         custo_corretagem, custo_slippage,carteira_inicial, carteira_pendente_inicial ,margem_troca=0.20, 
                         alpha_ema=0.20, limite_min_por_ativo=0.03, aliquota_cdi=0.225, anos_memoria=4, capital_inicial=100000):
  
    """"
        Simulação do pregão dia a dia de forma isolada.
    """

    historico_x =  list(brain.X_treino_scaled)
    X_teste_cru = df_features_teste.loc[df_teste_acoes.index, colunas_hmm].values
    max_passos_memoria = anos_memoria * 252

    carteira_atual, carteira_pendente = carteira_inicial, carteira_pendente_inicial #somente pra primeira janela
    dias_restantes, dias_holding = 0, tempo_regime
    prob_suavizada = None
    logs_diarios = []
    logs_backtest = []
    logs_carteiras = []

    for i in range(1, len(df_teste_acoes)):

        data_atual = df_teste_acoes.index[i]
        X_scaled_ontem = brain.scaler.transform(X_teste_cru[i - 1].reshape(1, -1)).flatten()
        historico_x.append(X_scaled_ontem)
        janela_inferencia = np.array(historico_x[-max_passos_memoria:])
        prob_s_ontem = brain.modelo.predict_proba(np.array(janela_inferencia))[-1]
        estado_hmm_atual = np.argmax(prob_s_ontem)

        prob_regimes_hoje = np.zeros(len(brain.estados_possiveis))
        for estado_hmm_idx, prob in enumerate(prob_s_ontem):
            regime = brain.mapa_risco[estado_hmm_idx]
            prob_regimes_hoje[regime] = prob

        #estado_cru_mais_provavel = np.argmax(prob_estados_hoje_cru)
        regime_atual = np.argmax(prob_regimes_hoje)
        #P3 = np.linalg.matrix_power(brain.transmat_original, 3)
        #prob_estado_futuro = P3[estado_hoje]

        P3_ordenada = np.linalg.matrix_power(brain.transmat_ordenada,3)
        prob_regimes_futuro = prob_regimes_hoje @ P3_ordenada

        if prob_suavizada is None:
            prob_suavizada = prob_regimes_futuro.copy()
        else: 
            prob_suavizada = alpha_ema * prob_regimes_futuro + (1 - alpha_ema) * prob_suavizada

        
        soma_prob = np.sum(prob_suavizada)
        if soma_prob > 0:
            prob_suavizada /= soma_prob
        
        pesos_sinteticos = {ativo: 0.0 for ativo in colunas_operacao}
        reward_sintetico = 0.0

        for s in brain.estados_possiveis:
            nome_carteira_s = brain.politica_otima.get(s, "100_CDI")
            carteira_s = brain.carteiras.get(nome_carteira_s, {})

            prob_s = prob_suavizada[s]

            for ativos in colunas_operacao:
                pesos_sinteticos[ativos] += carteira_s.get(ativos, 0.0) * prob_s

            reward_sintetico += brain.recompensas.get((s, nome_carteira_s), 0.0) * prob_s

        peso_residual_cortado = 0.0 #limpamos o que ficaria uma pequena porcentagem de caixa em um cenario 2% de peso
        for ativo, peso in pesos_sinteticos.items():
            if ativo != 'CDI' and peso < limite_min_por_ativo:
                peso_residual_cortado += peso
                pesos_sinteticos[ativo] = 0.0

        pesos_sinteticos['CDI'] += peso_residual_cortado
        brain.carteiras['Alvo_Suavizado'] = pesos_sinteticos.copy()
        brain.politica_otima[99] = 'Alvo_Suavizado'
        brain.recompensas[(regime_atual, 'Alvo_Suavizado')] = reward_sintetico

        if isinstance(custo_slippage, pd.DataFrame):
            pesos_atuais = portfolio.pesos().drop("Clearing", errors="ignore").to_dict()
            ativos_atuais_reais = [a for a,w in pesos_atuais.items() if w > 0.001 and a not in ["CDI", "Clearing"]]
            ativos_alvo_reais = [a for a,w in pesos_sinteticos.items() if w > 0.001 and a not in ["CDI", "Clearing"]]
            ativos_operados = list(set(ativos_atuais_reais) | set(ativos_alvo_reais))
            if ativos_operados and data_atual in custo_slippage.index:
                slippage_hoje = custo_slippage.loc[data_atual, ativos_operados].mean()
                custo_slippage_dia = min(slippage_hoje, 0.05)
            else:
                custo_slippage_dia = 0.0010
        else:
            custo_slippage_dia = custo_slippage
            

        if len(logs_backtest) == 0:
            patrimonio_anterior = portfolio.patrimonio_total()
        else:
            patrimonio_anterior = logs_backtest[-1]["Patrimonio"]

        resultado = executar_estrategia(
            portfolio=portfolio,
            retornos_dia=df_teste_acoes.iloc[i].to_dict(),
            estado_hoje=regime_atual,
            estado_futuro=99, 
            politica_otima=brain.politica_otima,
            recompensas=brain.recompensas,
            carteiras=brain.carteiras,
            carteira_atual=carteira_atual,
            carteira_pendente=carteira_pendente,
            dias_restantes=dias_restantes,
            dias_holding=dias_holding,
            holding_minimo=tempo_regime,
            margem_troca=margem_troca,
            aliquota_cdi=aliquota_cdi,
            custo_corretagem=custo_corretagem,
            custo_slippage=custo_slippage_dia, #custo_slippage
            reward_sintetico=reward_sintetico
        )
        patrimonio_atual = resultado["snapshot"]["Patrimonio"]
        retorno_modelo_dia = (patrimonio_atual / patrimonio_anterior) - 1

        retorno_benchmark_dia = df_teste_acoes.iloc[i]["Retorno_Ibov"]
        raw_cdi = df_teste_acoes.iloc[i].get("CDI", 0.0)
        retorno_taxa_cdi = (1 + raw_cdi / 100) ** (1 / 252) - 1 if raw_cdi > 0.005 else raw_cdi


        retorno_bench_hibrido_dia = benchmark_hibrido(retorno_benchmark_dia, retorno_taxa_cdi)
        retorno_bench_hibrido_dinamico = atribuicao_benchmark_dinamico(retorno_benchmark_dia, retorno_taxa_cdi, portfolio.exposicao())


        portfolio = resultado["portfolio"]
        carteira_atual = resultado["carteira_atual"]
        carteira_pendente = resultado["carteira_pendente"]
        dias_restantes = resultado["dias_restantes"]
        dias_holding = resultado["dias_holding"]
        reward_atual = resultado["reward_atual"]
        reward_nova = resultado["reward_nova"]
        acao_escolhida = resultado["acao_escolhida"]
        executou = resultado["executou"]


        logs_carteiras.append({
            'Data': data_atual,
            'evento': resultado["evento"],
            'Reward_delta': reward_nova - reward_atual,
            'Margem_exigida': resultado["margem_exigida"],
            'Troca_Aprovada': acao_escolhida != carteira_atual and reward_nova > reward_atual * (1 + margem_troca),
            "Regime_Futuro_Provavel": np.argmax(prob_suavizada),
            "Prob_Regimes": prob_suavizada.copy(),
            "Pesos_Alvo_Suavizado": pesos_sinteticos.copy(),
            'Carteira_Atual': carteira_atual,
            'Carteira_Pendente': carteira_pendente                    
        })

        resultado_log = {k: v for k, v in resultado.items() if k != "portfolio"}

        logs_diarios.append({
            "Data": data_atual,
            **resultado_log,
            "Regime_Macro": regime_atual,
            "Nome_Regime": brain.regimes[regime_atual]["nome"],
        })

        capital_clearing = resultado['snapshot']['Patrimonio'] - resultado['snapshot']['CDI'] - resultado['snapshot']['Capital_Acoes']

        logs_backtest.append({
            "Data": data_atual,
            "Estado_HMM": estado_hmm_atual,
            "Regime_Macro": regime_atual,
            "Nome_Regime": brain.regimes[regime_atual]["nome"],
            "Regime_Metrica": brain.metricas_regime[regime_atual],
            "Regime_Futuro": np.argmax(prob_suavizada),
            "Prob_Regime_Futuro": np.max(prob_suavizada),
            "Estado_Futuro": 99,
            "Evento": resultado["evento"],
            "Carteira_Atual": carteira_atual,
            "Carteira_Pendente": carteira_pendente,
            "Alocacao": acao_escolhida,
            "Executou": executou,
            "Reward_atual": reward_atual,
            "Reward_nova": reward_nova,
            "Patrimonio": round(resultado["snapshot"]["Patrimonio"], 4),
            "Capital_CDI": round(resultado["snapshot"]["CDI"], 4),
            "Capital_Acoes": round(resultado['snapshot']['Capital_Acoes'], 4),
            "Exposicao_Acoes": round(portfolio.exposicao(), 4),
            "Retorno_Modelo": retorno_modelo_dia,
            "Retorno_CDI": retorno_taxa_cdi,
            "Retorno_Benchmark": retorno_benchmark_dia,
            "Retorno_Benchmark_Hibrido": retorno_bench_hibrido_dia,
            "Retorno_Benchmark_Hibrido_Dinamico": retorno_bench_hibrido_dinamico,
            "Capital_Preso_Clearing": round(capital_clearing, 4),
            "Custo_Transacao": round(resultado["Custo_Transacao"], 4),
            "Custo_Slippage": round(resultado["Custo_Slippage"], 4),
            "Custo_Corretagem": round(resultado["Custo_Corretagem"], 4),
            **resultado["pesos"],
        })

    return portfolio, logs_backtest, logs_carteiras ,carteira_atual, carteira_pendente