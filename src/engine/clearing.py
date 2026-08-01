class ClearingHouse:
    """
    Entidade responsável por processar a custódia temporária, 
    ajustes físicos de liquidação e a fila de compensação diária (D+2).
    """

    @classmethod
    def processar_fila_diaria(cls, portfolio):
        """
        Diminui os dias restantes de cada lote preso na fila de compensação física.
        Ao chegar a zero dias, o saldo financeiro retorna líquido para o caixa do CDI.
        
        Essa função deve rodar no início de cada pregão diário (antes da marcação a mercado).
        """
        fila_residual = []
        for lote in portfolio.fila_clearing:
            lote["dias_restantes"] -= 1

            if lote['dias_restantes'] <= 0:
                portfolio.cdi_saldo += lote['valor']
                portfolio.cdi_custo += lote['valor']
            else:
                fila_residual.append(lote)

        portfolio.fila_clearing = fila_residual
        
        portfolio.patrimonio = portfolio.patrimonio_total()