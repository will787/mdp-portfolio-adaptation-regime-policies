import numpy as np
import sys
from src.engine.strategy import executar_strategy
from src.engine.benchmark import benchmark_hibrido



def simular_janela_teste(portfolio, df_teste_acoes, df_features_teste, brain, colunas_hmm, colunas_operacao, tempo_regime,
                         custo_corretagem, custo_slippage, carteira_inicial, carteira_pendente_inicial ,margem_troca=0.20, alpha_ema=0.20):
  
    """"
        Simulação do pregão dia a dia de forma isolada.
    """

    historico_x =  list(brain.X_treino_scaled)
    X_teste_cru = df_features_teste.loc[df_teste_acoes.index, colunas_hmm].values

    carteira_atual, carteira_pendente = carteira_inicial, carteira_pendente_inicial #somente pra primeira janela
    dias_restantes, dias_holding = 0, tempo_regime
    prob_suavizada = None
    logs_diarios = []
    logs_backtest = []
    logs_carteiras = []
    prob_estados_hoje_ordenado = None

    mapa_risco_rev = {v: k for k, v in brain.mapa_risco.items()}
    for i in range(1, len(df_teste_acoes)):

        data_atual = df_teste_acoes.index[i]

        X_scaled_ontem = brain.scaler.transform(X_teste_cru[i - 1].reshape(1, -1)).flatten()
        historico_x.append(X_scaled_ontem)

        prob_s_ontem = brain.modelo.predict_proba(np.array(historico_x))[-1]

        prob_estados_hoje_ordenado = np.zeros(len(brain.estados_possiveis))
        for estado_cru_idx, prob in enumerate(prob_s_ontem):
            nivel_risco = brain.mapa_risco[estado_cru_idx]
            prob_estados_hoje_ordenado[nivel_risco] = prob

        #estado_cru_mais_provavel = np.argmax(prob_estados_hoje_cru)
        estado_hoje = np.argmax(prob_estados_hoje_ordenado)
        #P3 = np.linalg.matrix_power(brain.transmat_original, 3)
        #prob_estado_futuro = P3[estado_hoje]

        P3_ordenada = np.linalg.matrix_power(brain.transmat_ordenada,3)
        prob_estado_futuro = np.dot(prob_estados_hoje_ordenado, P3_ordenada)

        if prob_suavizada is None:
            prob_suavizada = prob_estado_futuro.copy()
        else: 
            prob_suavizada = alpha_ema * prob_estado_futuro + (1 - alpha_ema) * prob_suavizada

        
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
            if ativo != 'CDI' and peso < 0.02:
                peso_residual_cortado += peso
                pesos_sinteticos[ativo] = 0.0

        pesos_sinteticos['CDI'] += peso_residual_cortado
        brain.carteiras['Alvo_Suavizado'] = pesos_sinteticos
        brain.politica_otima[99] = 'Alvo_Suavizado'
        brain.recompensas[(estado_hoje, 'Alvo_Suavizado')] = reward_sintetico

        resultado = executar_strategy(
            portfolio=portfolio,
            retornos_dia=df_teste_acoes.iloc[i].to_dict(),
            estado_hoje=estado_hoje,
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
            aliquota_cdi=0.225,
            custo_corretagem=custo_corretagem,
            custo_slippage=custo_slippage
        )

        patrimonio_anterior = portfolio.patrimonio if i == 1 else logs_backtest[-1]["Patrimonio"]
        retorno_modelo_dia = (resultado["snapshot"]["Patrimonio"] / patrimonio_anterior) - 1
        retorno_benchmark_dia = df_teste_acoes.iloc[i]["Retorno_Ibov"]
        retorno_taxa_cdi = df_teste_acoes.iloc[i].get('CDI', 0.0) #pct_change diario

        retorno_bench_hibrido_dia = benchmark_hibrido(retorno_benchmark_dia, retorno_taxa_cdi)


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
            'Prob_Estado_Futuro': np.max(prob_estado_futuro),
            'Carteira_Atual': carteira_atual,
            'Carteira_Pendente': carteira_pendente                    
        })

        resultado_log = {k: v for k, v in resultado.items() if k != "portfolio"}

        logs_diarios.append({
            "Data": data_atual,
            **resultado_log,
            "Regime_Macro": estado_hoje
        })

        capital_clearing = resultado['snapshot']['Patrimonio'] - resultado['snapshot']['CDI'] - resultado['snapshot']['Capital_Acoes']

        logs_backtest.append({
            "Data": data_atual,
            "Regime_Macro": estado_hoje,
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
            "Retorno_Benchmark": retorno_benchmark_dia,
            "Retorno_Benchmark_Hibrido": retorno_bench_hibrido_dia,
            "Capital_Preso_Clearing": round(capital_clearing, 4),
            "Custo_Transacao": round(resultado["Custo_Transacao"], 4),
            "Custo_Slippage": round(resultado["Custo_Slippage"], 4),
            "Custo_Corretagem": round(resultado["Custo_Corretagem"], 4),
            **resultado["pesos"]
        })

    return portfolio, logs_backtest, logs_carteiras ,carteira_atual, carteira_pendente