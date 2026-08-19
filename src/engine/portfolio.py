import numpy as np
import pandas as pd 


class Portfolio:
    """""
        Gerenciamento físico do balanço de ativos, caixa livre (CDI).
        fila de liquidação em D+2 e regras tributação de renda fixa (0,225)
    """

    def __init__(self, capital_inicial, ativos):
        self.ativos = {ativo: 0.0 for ativo in ativos if ativo != "CDI"}
        self.cdi_saldo = capital_inicial
        self.cdi_custo  = capital_inicial
        self.fila_clearing = []
        self.patrimonio = capital_inicial
        self.total_imposto_pago = 0.0

    def patrimonio_total(self):
        valor_ativos = sum(self.ativos.values())
        valor_retido_clearing = sum(lote['valor'] for lote in self.fila_clearing)
        return self.cdi_saldo + valor_ativos + valor_retido_clearing

    def pesos(self, ativos_filtro=None):
        patr = self.patrimonio_total()
        if patr <= 0:
            return pd.Series(dtype=float)

        pesos_dict = {ativo: valor/patr for ativo, valor in self.ativos.items()}
        pesos_dict['CDI'] = self.cdi_saldo / patr

        valor_retido_clearing = sum(lote['valor'] for lote in self.fila_clearing)
        pesos_dict['Clearing'] = valor_retido_clearing / patr

        if ativos_filtro is not None:
            for ativo in ativos_filtro:
                pesos_dict.setdefault(ativo, 0.0)

        return pd.Series(pesos_dict)

    def exposicao(self): #expo em acoes
        patr = self.patrimonio_total()
        if patr <= 0:
            return 0
        return sum(self.ativos.values())/patr

    def marcar_a_mercado(self, retornos):
        """"
            Aplica a oscilação de mercado sobre as posições atuais
            As ações sofrem marcação imediata de fechamento
            o CDI rende a taxa de hoje, mas o crédito só entra para o saldo do proximo dia. 
        """
        
        for ativo in self.ativos:
            self.ativos[ativo] *= (1 + retornos.get(ativo, 0.0))

        taxa_cdi_hoje = retornos.get("CDI", 0.0)
        rendimento_cdi_dia = self.cdi_saldo * taxa_cdi_hoje
        self.cdi_saldo += rendimento_cdi_dia
        
        self.patrimonio = self.patrimonio_total()

    def processar_clearing_diario(self):
        """"
            Atualiza a fila de liquidação física D+2 liberando recursos para o caixa livre.
        """

        fila_residual = []

        for lote in self.fila_clearing:
            lote["dias_restantes"] -= 1

            if lote['dias_restantes'] <= 0:
                self.cdi_saldo += lote['valor']
                self.cdi_custo += lote['valor']
            else:
                fila_residual.append(lote)

        self.fila_clearing = fila_residual
        self.patrimonio = self.patrimonio_total()

    def rebalancear(self, pesos_alvo, aliquota_cdi):
        patrimonio_inicial = self.patrimonio_total()
        pesos_atuais = self.pesos()

        for ativo in pesos_atuais.index:
            if ativo not in pesos_alvo:
                pesos_alvo[ativo] = 0.0

        for ativo, valor_atual in list(self.ativos.items()):
            peso_desejado = pesos_alvo.get(ativo, 0.0)
            valor_desejado = patrimonio_inicial * peso_desejado

            if valor_atual > valor_desejado:
                venda = valor_atual - valor_desejado
                self._vender_ativo(ativo, venda)
                self.fila_clearing.append({"valor": venda, "dias_restantes": 2})

        patrimonio_dinamico = self.patrimonio_total()

        for ativo, peso in pesos_alvo.items():
            if ativo in ["CDI", "Clearing"]:
                continue

            patrimonio_atualizado = self.patrimonio_total()
            atual = self.ativos.get(ativo, 0.0)
            desejado = patrimonio_atualizado * peso

            if desejado > atual:
                compra_necessaria = desejado - atual
                if self.cdi_saldo <= 1e-4:
                    continue

                saque_bruto = min(compra_necessaria,  self.cdi_saldo)
                liquido,imposto = self.resgatar_cdi(saque_bruto, aliquota_cdi)

                if liquido > 0:
                    self._comprar_ativo(ativo, liquido)

        self.patrimonio = self.patrimonio_total()

    def _comprar_ativo(self, ativo, valor):
            if valor <= 0: 
                return
            self.ativos[ativo] = self.ativos.get(ativo, 0.0) + valor

    def _vender_ativo(self, ativo, valor):
            atual = self.ativos.get(ativo, 0.0)
            valor = min(valor, max(0.0, atual := atual))
            novo = atual - valor

            if novo <= 1e-4:
                self.ativos.pop(ativo, None)
            else:
                self.ativos[ativo] = novo

    def resgatar_cdi(self, valor_resgate, aliquota):
            saldo = self.cdi_saldo
            custo = self.cdi_custo

            if saldo <= 0 or valor_resgate <= 0:
                return 0.0, 0.0

            valor_resgate = min(valor_resgate, saldo)
            rendimento = max(0, saldo - custo)

            proporcao = valor_resgate / saldo
            rendimento_resgatado = rendimento * proporcao       
            imposto = rendimento_resgatado * aliquota
            liquido = valor_resgate - imposto

            #deducao fisica do caixa livre
            self.cdi_saldo -= valor_resgate
            self.cdi_custo = max(0.0, self.cdi_custo - custo * proporcao)

            self.total_imposto_pago += imposto
            return liquido, imposto

    def snapshot(self):
        valor_retido_clearing = sum(lote['valor'] for lote in self.fila_clearing)

        return {
            "Patrimonio": self.patrimonio,
            "Ativos": self.ativos.copy(),
            "CDI": self.cdi_saldo,
            "Capital_Acoes": sum(self.ativos.values()),
            "Capital_Preso_Clearing": valor_retido_clearing,
            "Exposicao": self.pesos().to_dict(),
            "Quantidade_Ativos": len(self.ativos)
        }