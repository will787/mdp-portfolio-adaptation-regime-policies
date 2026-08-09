import numpy as np
import pandas as pd
from src.engine.portfolio import Portfolio
from src.engine.order_manager import OrderManager
from src.engine.clearing import ClearingHouse
from src.engine.cost_calculator import calcular_turnover, calcular_custo_transicao

def executar_estrategia(portfolio: Portfolio,retornos_dia: dict,estado_hoje: int, estado_futuro: int, politica_otima: dict,
                      recompensas: dict,carteiras: dict,carteira_atual: str, carteira_pendente: str,dias_restantes: int,
                      dias_holding: int,holding_minimo: int,margem_troca: float,aliquota_cdi: float,
                      custo_corretagem: float,custo_slippage: float, reward_sintetico: float) -> dict:
    """"
        Orquestração do ciclo diário de execução da cartiera: liquidação de caixa,
        avaliação de ordens de Bellman, marcação a mercado e rebalanceamentos
    """

    
    ClearingHouse.processar_fila_diaria(portfolio)

    custo_total_pct = 0.0
    custo_corr_dia = 0.0
    custo_slippage_dia = 0.0

    acao_escolhida = politica_otima[estado_futuro]
    reward_atual = recompensas[(estado_hoje, carteira_atual)]
    reward_nova = reward_sintetico

    portfolio.marcar_a_mercado(retornos_dia)
    manager = OrderManager(holding_minimo, margem_troca)
    status_ordem = manager.avaliar_sinal_diario(
        portfolio,
        acao_escolhida,
        carteira_atual,
        carteira_pendente,
        dias_restantes,
        dias_holding,
        reward_atual,
        reward_nova,
        carteiras
    )


    if status_ordem["executou"]:
        pesos_finais = carteiras[status_ordem["carteira_atual"]]
        pesos_atuais = portfolio.pesos().drop("Clearing", errors="ignore").to_dict()

        turnover = calcular_turnover(pesos_atuais, pesos_finais)
        custo_transicao = calcular_custo_transicao(turnover, custo_corretagem, custo_slippage)


        custo_total_pct = custo_transicao["total_pct"]
        custo_corr_dia = custo_transicao["corretagem_pct"]
        custo_slippage_dia = custo_transicao["slippage_pct"]

        if custo_total_pct > 0:
            financeiro_custo = portfolio.patrimonio * custo_total_pct
            portfolio.cdi_saldo -= financeiro_custo
            portfolio.cdi_custo = max(0.0, portfolio.cdi_custo - financeiro_custo)
            portfolio.patrimonio = portfolio.patrimonio_total()


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
        "Custo_Transacao": custo_total_pct,
        "Custo_Slippage": custo_slippage_dia,
        "Custo_Corretagem": custo_corr_dia
    }
    

    