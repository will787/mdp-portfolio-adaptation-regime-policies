import numpy as np
import pandas as pd 

class Portfolio:
    """
    Gerenciamento físico do balanço de ativos, caixa livre (CDI),
    fila de liquidação em D+2 e regras de tributação de renda fixa (alíquota padrão 0,225).
    """

    def __init__(self, capital_inicial, ativos):
        self.ativos = {ativo: 0.0 for ativo in ativos if ativo != "CDI"}
        self.cdi_saldo = float(capital_inicial)
        self.cdi_custo = float(capital_inicial)
        self.fila_clearing = []
        self.patrimonio = float(capital_inicial)
        self.total_imposto_pago = 0.0

    def patrimonio_total(self):
        valor_ativos = sum(self.ativos.values())
        valor_retido_clearing = sum(lote['valor'] for lote in self.fila_clearing)
        return self.cdi_saldo + valor_ativos + valor_retido_clearing

    def pesos(self, ativos_filtro=None):
        patr = self.patrimonio_total()
        if patr <= 0:
            return pd.Series(dtype=float)

        pesos_dict = {ativo: valor / patr for ativo, valor in self.ativos.items()}
        pesos_dict['CDI'] = self.cdi_saldo / patr

        valor_retido_clearing = sum(lote['valor'] for lote in self.fila_clearing)
        pesos_dict['Clearing'] = valor_retido_clearing / patr

        if ativos_filtro is not None:
            for ativo in ativos_filtro:
                pesos_dict.setdefault(ativo, 0.0)

        return pd.Series(pesos_dict)

    def exposicao(self):
        """Exposição percentual total em ações (renda variável)."""
        patr = self.patrimonio_total()
        if patr <= 0:
            return 0.0
        return sum(self.ativos.values()) / patr

    def marcar_a_mercado(self, retornos):
        """
        Aplica a oscilação de mercado sobre as posições atuais.
        As ações sofrem marcação imediata de fechamento e o CDI rende a taxa do dia.
        """
        for ativo in list(self.ativos.keys()):
            self.ativos[ativo] *= (1.0 + retornos.get(ativo, 0.0))

        taxa_cdi_hoje = retornos.get("CDI", 0.0)
        rendimento_cdi_dia = self.cdi_saldo * taxa_cdi_hoje
        self.cdi_saldo += rendimento_cdi_dia
        self.patrimonio = self.patrimonio_total()

    def processar_clearing_diario(self):
        """Atualiza a fila de liquidação física D+2 liberando recursos para o caixa livre."""
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

    def debitar_custos_operacionais(self, custo_financeiro):
        """Deduz custos de corretagem e slippage do caixa."""
        if custo_financeiro <= 0:
            return
        self.cdi_saldo -= custo_financeiro
        self.cdi_custo = max(0.0, self.cdi_custo - custo_financeiro)
        self.patrimonio = self.patrimonio_total()

    def rebalancear(self, pesos_alvo, aliquota_cdi):
        patrimonio_inicial = self.patrimonio_total()
        pesos_atuais = self.pesos()

        for ativo in pesos_atuais.index:
            if ativo not in pesos_alvo:
                pesos_alvo[ativo] = 0.0

        # 1. Execução de Vendas (Gera fila de Clearing D+2)
        for ativo, valor_atual in list(self.ativos.items()):
            peso_desejado = pesos_alvo.get(ativo, 0.0)
            valor_desejado = patrimonio_inicial * peso_desejado

            if valor_atual > valor_desejado:
                venda = valor_atual - valor_desejado
                self._vender_ativo(ativo, venda)
                self.fila_clearing.append({"valor": venda, "dias_restantes": 2})

        # 2. Execução de Compras (Limitadas ao CDI livre no momento)
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

                saque_bruto = min(compra_necessaria, self.cdi_saldo)
                liquido, imposto = self.resgatar_cdi(saque_bruto, aliquota_cdi)

                if liquido > 0:
                    self._comprar_ativo(ativo, liquido)

        self.patrimonio = self.patrimonio_total()

    def _comprar_ativo(self, ativo, valor):
        if valor <= 0: 
            return
        self.ativos[ativo] = self.ativos.get(ativo, 0.0) + valor

    def _vender_ativo(self, ativo, valor):
        atual = self.ativos.get(ativo, 0.0)
        valor = min(valor, max(0.0, atual))
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
        rendimento = max(0.0, saldo - custo)

        proporcao = valor_resgate / saldo
        rendimento_resgatado = rendimento * proporcao       
        imposto = rendimento_resgatado * aliquota
        liquido = valor_resgate - imposto

        self.cdi_saldo -= valor_resgate
        self.cdi_custo = max(0.0, self.cdi_custo - custo * proporcao)

        self.total_imposto_pago += imposto
        return liquido, imposto

    def snapshot(self):
        valor_retido_clearing = sum(lote['valor'] for lote in self.fila_clearing)
        return {
            "Patrimonio": round(self.patrimonio, 4),
            "Ativos": {k: round(v, 4) for k, v in self.ativos.items()},
            "CDI": round(self.cdi_saldo, 4),
            "Capital_Acoes": round(sum(self.ativos.values()), 4),
            "Capital_Preso_Clearing": round(valor_retido_clearing, 4),
            "Exposicao": self.pesos().to_dict(),
            "Quantidade_Ativos": len(self.ativos)
        }