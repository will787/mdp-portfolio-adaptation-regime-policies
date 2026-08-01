import numpy as np
import pandas as pd
import scipy.optimize as sco


def chute_score(score):
    score = score.clip(lower=0)
    if score.sum() == 0:
        chute = np.ones(len(score))
    else:
        chute = score.values
    chute = chute / chute.sum()
    return chute

def criar_carteiras_por_regime(df_treino_acoes, estados_possiveis, ativos_vivos, ativos_risco, 
                               colunas_operacao, tempo_regime, metrica_otimizacao, limite_max_por_ativo):
    """"
        Filtra as top 30 ações baseadas em utilidade e risco por regime do HMM
        resolvendo a otimização de pesos de forma isolada. 
    """

    acoes_disponiveis_dinamico = {}

    acoes_disponiveis_dinamico["100_CDI"] = {ativo: 0.0 for ativo in colunas_operacao}
    acoes_disponiveis_dinamico["100_CDI"]["CDI"] = 1.0

    composicoes_fixas = {
        "80_CDI": 0.80,
        "60_CDI": 0.60,
        "40_CDI": 0.40,
        "20_CDI": 0.20
    }
    peso_eq = 1 / len(ativos_vivos) if len(ativos_vivos) else 0

    for nome_carteira, peso_cdi in composicoes_fixas.items():
        acoes_disponiveis_dinamico[nome_carteira] = {ativo: 0.0 for ativo in colunas_operacao}
        acoes_disponiveis_dinamico[nome_carteira]["CDI"] = peso_cdi
        peso_risco_rest = 1 - peso_cdi

        for ativo in ativos_vivos:
            acoes_disponiveis_dinamico[nome_carteira][ativo] = peso_eq * peso_risco_rest


    for s in estados_possiveis:
        dias_estado = df_treino_acoes[df_treino_acoes['Estado_HMM'] == s]

        if len(dias_estado) > tempo_regime:
            retornos_ativos = dias_estado[ativos_vivos]
            selic_media_regime = dias_estado['CDI'].mean()


            ret = (1 + retornos_ativos.mean()) ** 252 - 1
            vol = retornos_ativos.std() * np.sqrt(252)
            drag = (vol**2) / 2

            ret_negativo = retornos_ativos[retornos_ativos < 0]
            downside = np.sqrt((ret_negativo ** 2).mean()) * np.sqrt(252)
            score = (ret -0.5 * drag - 0.3 * downside)

            topx = score.nlargest(20).index
            retornos_fatiados = retornos_ativos[topx]
            score_fatiado = score[topx]


            chute = chute_score(score_fatiado)


            metrica_regime = metrica_otimizacao
            if metrica_otimizacao == 'adaptativo':
                metrica_regime = ['omega', 'sortino', 'cvar', 'min_vol'][s]

            carteira_otima_crua, _ = otimizar_carteira_por_regime(
                retornos_estado = retornos_fatiados,
                taxa_livre_risco_diaria = selic_media_regime,
                metrica=metrica_regime,
                limite_max_por_ativo = limite_max_por_ativo,
                chute_inicial = chute
            )


            carteira_completa = {ativo: 0.0 for ativo in colunas_operacao}
            for ativo, peso in carteira_otima_crua.items():
                carteira_completa[ativo] = peso

            carteira_completa['CDI'] = carteira_completa.get('CDI', 0.0)


            acoes_disponiveis_dinamico[f'Otima_Regime{s}'] = carteira_completa

        else:
            acoes_disponiveis_dinamico[f'Otima_Regime{s}'] = acoes_disponiveis_dinamico['100_CDI']

    nomes_acoes_dinamico = list(acoes_disponiveis_dinamico.keys())
    return acoes_disponiveis_dinamico, nomes_acoes_dinamico


def otimizar_carteira_por_regime(retornos_estado, taxa_livre_risco_diaria, metrica, limite_max_por_ativo, chute_inicial, limite_min_por_ativo=0.02):
    """"
        Otimizador númerico SQSQP (Markovitz / Sortino / Calmar Adaptativo)
        Aplicamos a capitalização para anualizar retornos e volatilidade
        Métricas suportadas, 'omega' (Bear), 'sortino' (Transição), 'cvar' (Correção), min_vol
    """

    num_ativos = len(retornos_estado.columns)

    taxa_livre_de_risco_anual = (1 + taxa_livre_risco_diaria) ** 252 -1

    if (num_ativos * limite_min_por_ativo) > 1.0:
        limite_min_ajustado = 1.0 / num_ativos
        print(f'Limite mínimo: {limite_min_por_ativo*100} para {num_ativos} ativos excede 100%. Reduzindo mínimo para {limite_min_ajustado*100:.2f}')
        limite_min_por_ativo = limite_min_ajustado

    if limite_max_por_ativo < limite_min_por_ativo:
        limite_max_por_ativo = limite_min_por_ativo

    #limite_minimo_bolsa = 0.60 if metrica == 'min_vol' else 0.0

    def funcao_objetivo(pesos):
        retorno_diario_p = np.dot(retornos_estado, pesos)

        retorno_medio_anual = (1 + np.mean(retorno_diario_p)) ** 252 -1
        retorno_excesso_anual = retorno_medio_anual - taxa_livre_de_risco_anual

        if metrica == 'sortino':
            #sortino adaptado quando estava perfomando abaixo da taxa livre de risco
            retornos_negativos = retorno_diario_p[retorno_diario_p < taxa_livre_risco_diaria]
            if len(retornos_negativos) > 2:
                downside_vol_anual = np.sqrt(np.mean((retornos_negativos - taxa_livre_risco_diaria)**2)) * np.sqrt(252)
            else:
                downside_vol_anual = 1e-6

            if retorno_excesso_anual < 0:
                return -retorno_excesso_anual * (downside_vol_anual + 1e-6)
            return -retorno_excesso_anual / (downside_vol_anual + 1e-6)

        elif metrica == 'sharpe':
            vol_total_anual = np.std(retorno_diario_p) * np.sqrt(252)
            return -(retorno_excesso_anual) / (vol_total_anual + 1e-6)

        elif metrica == 'omega':
            ganhos = retorno_diario_p[retorno_diario_p > taxa_livre_risco_diaria] - taxa_livre_risco_diaria
            perdas = taxa_livre_risco_diaria - retorno_diario_p[retorno_diario_p < taxa_livre_risco_diaria]
            sum_ganhos = np.sum(ganhos) if len(ganhos) > 0 else 0.0
            sum_perdas = np.sum(perdas) if len(perdas) > 0 else 1e-6
            omega = sum_ganhos / (sum_perdas + 1e-6)
            return -omega

        elif metrica == 'cvar':
            alpha = 0.05
            var_95 = np.percentile(retorno_diario_p, alpha * 100)
            piores_retornos = retorno_diario_p[retorno_diario_p <= var_95]
            cvar_diario = np.mean(piores_retornos) if len(piores_retornos) > 0 else var_95
            cvar_anual =  abs(cvar_diario) * np.sqrt(252)

            if retorno_excesso_anual < 0:
                return -retorno_excesso_anual * (cvar_anual + 1e-6)
            return -retorno_excesso_anual / (cvar_anual + 1e-6)

        elif metrica == 'calmar':
            ret_ajustado = np.clip(retorno_diario_p, -0.99, None)
            cum_ret = np.cumprod(1 + ret_ajustado)
            peak = np.maximum.accumulate(cum_ret)
            drawdown = (cum_ret - peak)/ (peak+1e-8)
            max_drawdown = np.abs(np.min(drawdown))

            return -(retorno_excesso_anual) / (max_drawdown + 1e-6)

        elif metrica == 'min_vol':
            volatilidade_anual = np.std(retorno_diario_p) * np.sqrt(252)
            return volatilidade_anual

        return - retorno_excesso_anual

    restricoes = [{'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)}]

    limites = tuple((limite_min_por_ativo, limite_max_por_ativo) for _ in range(num_ativos))

    if chute_inicial is None:
        chute_inicial = np.array([1.0 / num_ativos] * num_ativos)

    resultado = sco.minimize(
        funcao_objetivo,
        chute_inicial,
        method='SLSQP',
        bounds=limites,
        constraints=restricoes,
        options = {
            "maxiter": 500,
            "ftol": 1e-9,
        }
    )

    pesos = pd.Series(resultado.x, index=retornos_estado.columns)
    pesos = pesos.apply(lambda w: w if w > limite_min_por_ativo else 0.0)

    if pesos.sum() > 0:
        pesos = pesos / pesos.sum()

    fator_alocacao_bolsa = calcular_exposicao_kelly(
        retornos_estado=retornos_estado,
        pesos=pesos,
        taxa_livre_risco_diaria=taxa_livre_risco_diaria,
        fracao_kelly=0.50,
        teto_exposicao=1.0
    )

    pesos_finais_acoes = pesos * fator_alocacao_bolsa

    carteira_otima = pesos_finais_acoes.to_dict()
    peso_total_alocado = pesos_finais_acoes.sum()
    carteira_otima['CDI'] = max(0.0, 1.0 - peso_total_alocado)

    return carteira_otima, peso_total_alocado


def calcular_metricas_bellman(estados_possiveis, df_treino_acoes, acoes_disponiveis_dinamico,
                              colunas_operacao, tempo_regime, metrica_otimizacao):

    """"
        Avalia a matriz de utilidade histórica de cada combinação de Estado e Carteira.
        Anualiza os prêmios de risco de forma geometrica real (Juros compostos) para alimentar a Equação de Bellman.
    """

    recompensas = {}
    metricas_recompensa = {}


    idx_cdi = colunas_operacao.index('CDI') if "CDI" in colunas_operacao else -1


    for s in estados_possiveis:
        dias_estado = df_treino_acoes[df_treino_acoes['Estado_HMM'] == s]

        if len(dias_estado) <= tempo_regime:
            for nome_acao in acoes_disponiveis_dinamico.keys():
                recompensas[(s, nome_acao)] = -1.0
                metricas_recompensa[(s, nome_acao)] = {
                    "reward": -1.0,
                    "retorno_anualizado": np.nan,
                    "retorno_excesso": np.nan,
                    "volatilidade": np.nan,
                    "metrica": None,
                    'drawdown': np.nan,
                    "peso_cdi": np.nan,
                    "num_ativos": np.nan
                }
            continue

        retornos_diario_estado = dias_estado[colunas_operacao].values

        cdi_diario_regime = retornos_diario_estado[:, idx_cdi]
        cdi_medio_diario = cdi_diario_regime.mean()
        taxa_livre_risco_anual = (1 + cdi_diario_regime.mean()) ** 252 -1

        for nome_acao, pesos in acoes_disponiveis_dinamico.items():
            pesos_array = np.array([
                pesos.get(ativo,0.0)
                for ativo in colunas_operacao
            ])

            retornos_portfolio_diario = np.dot(
                retornos_diario_estado,
                pesos_array
            )

            #anualização composto geometrica real do portfolio (CAGR)

            retorno_anualizado = (1 + retornos_portfolio_diario.mean()) ** 252 -1
            retorno_excesso = retorno_anualizado - taxa_livre_risco_anual
            vol_anual = np.std(retornos_portfolio_diario) * np.sqrt(252)

            ret_ajustado = np.clip(retornos_portfolio_diario, -0.99, None)

            cum_ret = np.cumprod(1 + ret_ajustado)
            peak = np.maximum.accumulate(cum_ret)
            drawdown = (cum_ret - peak) / (peak + 1e-8)
            max_drawdown = np.abs(np.min(drawdown))

            metrica_atual = metrica_otimizacao
            if metrica_otimizacao == 'adaptativo':
            
                metrica_atual = {
                    0: "omega",
                    1: "sortino",
                    2: "cvar",
                    3: "min_vol",
                }[s]

            if metrica_atual == "sortino":
                ret_negativo = retornos_portfolio_diario[retornos_portfolio_diario < 0]
                if len(ret_negativo):
                    downside = np.sqrt(np.mean(ret_negativo**2)) * np.sqrt(252)
                    recompensa = retorno_excesso / (downside + 1e-6)
                else:
                    recompensa = retorno_excesso

            elif metrica_atual == 'sharpe':
                recompensa = retorno_excesso / (vol_anual + 1e-6)

            elif metrica_atual == 'cvar':
                alpha = 0.05
                var_95 = np.percentile(retornos_portfolio_diario, alpha * 100)
                piores_retornos = retornos_portfolio_diario[retornos_portfolio_diario <= var_95]
                cvar_diario = np.mean(piores_retornos) if len(piores_retornos) > 0 else var_95
                cvar_anual = abs(cvar_diario) * np.sqrt(252)

                if retorno_excesso < 0:
                    recompensa = retorno_excesso * (cvar_anual + 1e-6)
                else:
                    recompensa = retorno_excesso / (cvar_anual + 1e-6)

            elif metrica_atual == 'omega':
                ganhos = retornos_portfolio_diario[retornos_portfolio_diario > cdi_medio_diario] - cdi_medio_diario
                perdas = cdi_medio_diario - retornos_portfolio_diario[retornos_portfolio_diario < cdi_medio_diario]
                sum_ganhos = np.sum(ganhos) if len(ganhos) > 0 else 0.0
                sum_perdas = np.sum(perdas) if len(perdas) > 0 else 1e-6
                recompensa = sum_ganhos / (sum_perdas + 1e-6)

            elif metrica_atual == 'calmar':
                recompensa = retorno_excesso / (max_drawdown + 1e-6)

            elif metrica_atual == 'min_vol':
                recompensa = -vol_anual
            else:
                recompensa = retorno_excesso

            recompensas[(s, nome_acao)] = recompensa
            metricas_recompensa[(s, nome_acao)] = {
                "reward": recompensa,
                "retorno_anualizado": retorno_anualizado,
                "retorno_excesso": retorno_excesso,
                "volatilidade": vol_anual,
                "drawdown": max_drawdown,
                "metrica": metrica_atual,
                "peso_cdi": pesos.get("CDI", 0),
                "num_ativos": np.sum(pesos_array > 0.001)
            }

    return recompensas, metricas_recompensa



def calcular_exposicao_kelly(retornos_estado, pesos, taxa_livre_risco_diaria, fracao_kelly=0.50, teto_exposicao=1.0):
    """
    Calcula a exposição global ótima da carteira usando o Critério de Kelly.
    
    Parâmetros:
    - retornos_estado: DataFrame/Array com retornos diários dos ativos do regime.
    - pesos: Pandas Series ou Array com os pesos relativos dos ativos (soma = 1.0).
    - taxa_livre_risco_diaria: Taxa livre de risco diária do período.
    - fracao_kelly: Multiplicador de risco (0.50 para Half-Kelly).
    - teto_exposicao: Limite máximo de alocação em bolsa (1.0 = 100%).
    
    Retorna:
    - fator_alocacao_bolsa (float): Percentual ótimo para expor em bolsa [0.0, teto_exposicao].
    """
    retorno_carteira_diario = np.dot(retornos_estado, pesos)

    retorno_medio_anual = np.mean(retorno_carteira_diario) * 252
    taxa_livre_anual = taxa_livre_risco_diaria * 252
    excesso_retorno_anual = retorno_medio_anual - taxa_livre_anual
    variancia_anual = np.var(retorno_carteira_diario) * 252

    if variancia_anual > 1e-8 and excesso_retorno_anual > 0:
        f_kelly_puro = excesso_retorno_anual / variancia_anual
        fator_alocacao_bolsa = f_kelly_puro * fracao_kelly
    else:
        fator_alocacao_bolsa = 0.0
        
    fator_alocacao_bolsa = np.clip(fator_alocacao_bolsa, 0.0, teto_exposicao)
        
    return float(fator_alocacao_bolsa)