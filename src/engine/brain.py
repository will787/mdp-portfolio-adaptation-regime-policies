import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from src.engine.bellman_equation import belmann_equation
from src.engine.wallets import criar_carteiras_por_regime, calcular_metricas_bellman


class MarketBrain:
    """"
        Encapsulamento da inteligência estatística da rodada.
        Treinamento do HMM cáclulo de scores, métricas de bellman e política ótima
    """

    def __init__(self, estados_possiveis=[0, 1, 2, 3], n_iter=1000, random_state=42):
        self.estados_possiveis = estados_possiveis
        self.n_iter = n_iter
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.modelo = hmm.GaussianHMM(n_components=len(estados_possiveis), covariance_type='full', n_iter=n_iter, random_state=random_state)
        self.mapa_risco = {}
        self.transmat_orignal = None
        self.transmat_ordenada = None
        self.transition_probs = {}
        self.carteiras = {}
        self.recompensas = {}
        self.metricas_recompensa = {}
        self.politica_otima = {}
        self.X_treino_scaled = None

    def treinar_e_otimizar(self, dados_treino_hmm, retornos_sinal, colunas_hmm, colunas_operacao,
                            tempo_regime, metrica_otimizacao, limite_max_por_ativo):

        df_treino_acoes = retornos_sinal.loc[dados_treino_hmm.index]
        ativos_risco = [col for col in colunas_operacao if col != 'CDI']
        ativos_vivos = [ativo for ativo in ativos_risco if df_treino_acoes[ativo].std() > 1e-6]

        X_treino = dados_treino_hmm[colunas_hmm].values
        self.X_treino_scaled = self.scaler.fit_transform(X_treino)

        matriz_persistencia_085 = np.array([
        [0.85, 0.12, 0.02, 0.01],  # Bull Market: alta persistência, com alguma chance de ir para Transição
        [0.08, 0.77, 0.12, 0.03],  # Transição: mais volátil, pode ir tanto para Bull quanto para Correção
        [0.02, 0.10, 0.78, 0.10],  # Correção: espaço intermediário de estresse
        [0.01, 0.02, 0.12, 0.85]   # Crise (Vol Extrema): alta inércia de fundo do poço, mas destrava se começar a recuperar
        ])

        self.modelo.transmat_ = matriz_persistencia_085
        self.modelo.init_params = 'smc'
        self.modelo.fit(self.X_treino_scaled)

        volatilidades = [self.modelo.means_[i][1] for i in range(len(self.estados_possiveis))]
        ordem_risco = np.argsort(volatilidades)
        self.mapa_risco = {estado_hmm: nivel_risco for nivel_risco, estado_hmm in enumerate(ordem_risco)}

        self.transmat_original = self.modelo.transmat_
        self.transition_probs = {
            (i, j): self.transmat_original[np.where(ordem_risco == i)[0][0]][np.where(ordem_risco == j)[0][0]]
            for i in self.estados_possiveis for j in self.estados_possiveis
        }                              

        n_states = len(self.estados_possiveis)
        self.transmat_ordenada = np.zeros((n_states, n_states))
        for i in self.estados_possiveis:
            for j in self.estados_possiveis:
                self.transmat_ordenada[i,j] = self.transition_probs[(i,j)]

        estado_treino = [self.mapa_risco[s] for s in self.modelo.predict(self.X_treino_scaled)]
        df_treino_acoes = df_treino_acoes.assign(Estado_HMM=estado_treino)


        self.carteiras, nomes_carteiras = criar_carteiras_por_regime(
            df_treino_acoes, self.estados_possiveis, ativos_vivos, ativos_risco, colunas_operacao, tempo_regime, metrica_otimizacao, limite_max_por_ativo
        )

        self.recompensas, self.metricas_recompensa = calcular_metricas_bellman(
            estados_possiveis=self.estados_possiveis, 
            df_treino_acoes=df_treino_acoes, 
            acoes_disponiveis_dinamico=self.carteiras,
            colunas_operacao=colunas_operacao,
            tempo_regime=tempo_regime,
            metrica_otimizacao=metrica_otimizacao
        )

        _, self.politica_otima = belmann_equation(
            self.estados_possiveis,
            nomes_carteiras,
            self.transition_probs,
            self.recompensas
        )