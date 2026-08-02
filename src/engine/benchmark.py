import numpy as np
import pandas as pd 


def benchmark_hibrido(ibov, cdi, peso_ibov=0.5, peso_cdi=0.5):
    """"
        misturamos a composição dos estados do benchmark.
        aplicando um peso de 50% do indice do ibovespa 50% do indice CDI.

        Os inputs devem estar em variações pra montarmos a estrutura de forma correta.
        Por seus respectivos pesos.

        Um benchmark hibrido estatico.
    """

    if (peso_ibov + peso_cdi) > 1.0:
        print('Peso da atribuição maior que 100%')
        return None

    bench = (ibov * peso_ibov) + (cdi * peso_cdi)

    return bench

def atribuicao_benchmark_dinamico(ibov, cdi, exposicao):
    """"
        Bench híbrido dinâmico utilizada para análise de perfomance attribution.

        Onde representa aproximadamente o alpha gerado pela seleção de ativos,
        dos componentes da estratégia.
    """
    exposicao = np.clip(exposicao, 0.0, 1.0)

    return exposicao * ibov + (1.0 - exposicao) * cdi