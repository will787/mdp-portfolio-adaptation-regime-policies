import pandas as pd

class OrderManager:
    """
    Gerenciador e validador do ciclo de vida de ordens do Bellman.
    Controla travas de Holding e o atraso físico de execução (Delay).
    """
    def __init__(self, holding_minimo: int = 21, margem_troca: float = 0.20):
        self.holding_minimo = holding_minimo
        self.margem_troca = margem_troca

    def avaliar_sinal_diario(self, portfolio, acao_escolhida, carteira_atual, carteira_pendente,
                             dias_restantes, dias_holding, reward_atual, reward_nova, carteiras) -> dict:
        """
        Determina o evento de trading do dia e gerencia o cronômetro de execução.
        """
        evento = "MANTER"
        executou = False
        margem_exigida = self.margem_troca

        # 1. TRAVA DE CARÊNCIA MÍNIMA (HOLDING)
        if dias_holding <= self.holding_minimo:
            dias_holding += 1
            evento = "HOLDING"
            return self._gerar_status(carteira_atual, carteira_pendente, dias_restantes, dias_holding, evento, executou, margem_exigida)

        # 2. SE NÃO HÁ ORDEM EM CURSO, VALIDA ENTRADA DE NOVAS PROPOSTAS
        if carteira_pendente is None:
            if acao_escolhida == carteira_atual:
                pesos_meta = carteiras[acao_escolhida]
                pesos_reais = portfolio.pesos()
                
                todos = set(pesos_reais.keys()).union(set(pesos_meta.keys()))
                desvio = sum(abs(pesos_reais.get(at, 0.0) - pesos_meta.get(at, 0.0)) for at in todos)

                if desvio > self.margem_troca:
                    carteira_pendente = acao_escolhida
                    dias_restantes = 3
                    evento = "ORDEM_CRIADA_REBALANCEAMENTO"
                else:
                    evento = "SEM_TROCA"
            else:
                # Compara recompensas de Bellman (Política Ótima vs Atual)
                if reward_nova > reward_atual * (1 + self.margem_troca):
                    margem_exigida = reward_atual * (1 + self.margem_troca)
                    carteira_pendente = acao_escolhida
                    dias_restantes = 3
                    evento = "ORDEM_CRIADA"
                else:
                    evento = "ORDEM_REJEITADA"

        # 3. GERENCIADOR DE ATRASO OPERACIONAL (CONTADOR DE 3 DIAS)
        if carteira_pendente is not None:
            dias_restantes -= 1
            if dias_restantes > 0:
                evento = "AGUARDANDO_EXECUCAO"
            else:
                carteira_atual = carteira_pendente
                carteira_pendente = None
                dias_holding = 0
                evento = "EXECUTOU_TROCA"
                executou = True

        return self._gerar_status(carteira_atual, carteira_pendente, dias_restantes, dias_holding, evento, executou, margem_exigida)

    def _gerar_status(self, c_atual, c_pendente, d_restantes, d_holding, ev, exec, m_exigida) -> dict:
        return {
            "carteira_atual": c_atual,
            "carteira_pendente": c_pendente,
            "dias_restantes": d_restantes,
            "dias_holding": d_holding,
            "evento": ev,
            "executou": exec,
            "margem_exigida": m_exigida
        }
