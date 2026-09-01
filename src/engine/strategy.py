import numpy as np
import pandas as pd
from src.engine.portfolio import Portfolio
from src.engine.order_manager import OrderManager
from src.engine.clearing import ClearingHouse
from src.engine.cost_calculator import calcular_turnover, calcular_custo_transicao

def executar_estrategia(portfolio: Portfolio, retornos_dia: dict, estado_hoje: int, estado_futuro: int, 
                        politica_otima: dict, recompensas: dict, carteiras: dict, carteira_atual: str, 
                        carteira_pendente: str, dias_restantes: int, dias_holding: int, holding_minimo: int, 
                        margem_troca: float, aliquota_cdi: float, custo_corretagem: float, 
                        custo_slippage: float, reward_sintetico: float) -> dict:
    """
    Orquestração do ciclo diário de execução da carteira: liquidação D+2,
    marcação a mercado, avaliação de ordens por Bellman, débito de custos em R$ e rebalanceamento.
    """

    ClearingHouse.processar_fila_diaria(portfolio)

    portfolio.marcar_a_mercado(retornos_dia)

    acao_escolhida = politica_otima[estado_futuro]
    reward_atual = recompensas.get((estado_hoje, carteira_atual), 0.0)
    reward_nova = reward_sintetico

    manager = OrderManager(holding_minimo, margem_troca)
    status_ordem = manager.avaliar_sinal_diario(
        portfolio=portfolio,
        acao_escolhida=acao_escolhida,
        carteira_atual=carteira_atual,
        carteira_pendente=carteira_pendente,
        dias_restantes=dias_restantes,
        dias_holding=dias_holding,
        reward_atual=reward_atual,
        reward_nova=reward_nova,
        carteiras=carteiras
    )

    custo_total_pct = 0.0
    custo_corr_pct = 0.0
    custo_slippage_pct = 0.0
    financeiro_custo = 0.0

    if status_ordem["executou"]:
        pesos_finais = carteiras[status_ordem["carteira_atual"]]
        pesos_atuais = portfolio.pesos().drop("Clearing", errors="ignore").to_dict()

        # Calcula Turnover e Taxas percentuais
        turnover = calcular_turnover(pesos_atuais, pesos_finais)
        custo_transicao = calcular_custo_transicao(turnover, custo_corretagem, custo_slippage)

        custo_total_pct = float(custo_transicao["total_pct"])
        custo_corr_pct = float(custo_transicao["corretagem_pct"])
        custo_slippage_pct = float(custo_transicao["slippage_pct"])

        if custo_total_pct > 0:
            financeiro_custo = portfolio.patrimonio_total() * custo_total_pct
            portfolio.debitar_custos_operacionais(financeiro_custo)

        portfolio.rebalancear(pesos_finais, aliquota_cdi)

    snap = portfolio.snapshot()
    pesos_finais_dia = portfolio.pesos().to_dict()

    return {
        "portfolio": portfolio,
        "evento": status_ordem["evento"],
        "carteira_atual": status_ordem["carteira_atual"],
        "carteira_pendente": status_ordem["carteira_pendente"],
        "dias_restantes": status_ordem["dias_restantes"],
        "dias_holding": status_ordem["dias_holding"],
        "reward_atual": reward_atual,
        "reward_nova": reward_nova,
        "margem_exigida": status_ordem["margem_exigida"],
        "acao_escolhida": acao_escolhida,
        "executou": status_ordem["executou"],
        "snapshot": snap,
        "pesos": pesos_finais_dia,
        "Custo_Transacao": round(financeiro_custo, 4),        # Valor em R$ debitado
        "Custo_Transacao_Pct": round(custo_total_pct, 6),    # Taxa percentual decimal
        "Custo_Slippage": round(portfolio.patrimonio_total() * custo_slippage_pct, 4),
        "Custo_Corretagem": round(portfolio.patrimonio_total() * custo_corr_pct, 4)
    }