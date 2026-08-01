import numpy as np 

def calcular_turnover(pesos_atuais: dict, pesos_alvo: dict) -> float:
    """"
        Calcula a taxa de giros de ativos (turnover) entre duas carteiras.
        Fórmula: (soma do desvio absoluto de todos os pesos) / 2
    """

    todos_ativos = set(pesos_atuais.keys()).union(set(pesos_alvo.keys()))

    desvio_total = sum(
        abs(pesos_alvo.get(ativo, 0.0) - pesos_atuais.get(ativo, 0.0))
        for ativo in todos_ativos
    )

    return desvio_total / 2


def calcular_custo_transicao(turnover:float, custo_corretagem:float, custo_slippage:float):
    """"
            Computa o impacto financeiro percentual com base na taxas e no giro realizado.
    """

    custo_total_pct = turnover * (custo_corretagem + custo_slippage)
    return {
        "total_pct": custo_total_pct,
        "corretagem_pct": turnover * custo_corretagem,
        "slippage_pct": turnover * custo_slippage
    }


