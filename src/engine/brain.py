import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from src.engine.bellman_equation import belmann_equation
from src.engine.wallets import criar_carteiras_por_regime, calcular_metricas_bellman

import warnings
from pandas.errors import PerformanceWarning
warnings.filterwarnings('ignore', category=PerformanceWarning)



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
        self.colunas_hmm = None
        self.dados_treino_original = None
        self.transition_probs = {}
        self.carteiras = {}
        self.recompensas = {}
        self.metricas_recompensa = {}
        self.politica_otima = {}
        self.X_treino_scaled = None
        self.relatorios_regimes = None
        self.metricas_regime = {0: "sharpe",1: "sortino",2: "cvar",3: "min_vol"}
        self.regimes = self.criar_regimes()

    def treinar_e_otimizar(self, dados_treino_hmm, retornos_sinal, colunas_hmm, colunas_operacao,
                            tempo_regime, metrica_otimizacao, limite_max_por_ativo, limite_min_por_ativo, numero_ativos, matriz_transicao):

        df_treino_acoes = retornos_sinal.loc[dados_treino_hmm.index]
        ativos_risco = [col for col in colunas_operacao if col != 'CDI']
        ativos_vivos = [ativo for ativo in ativos_risco if df_treino_acoes[ativo].std() > 1e-6]

        X_treino = dados_treino_hmm[colunas_hmm].values
        self.dados_treino_original = dados_treino_hmm[colunas_hmm].copy()
        self.X_treino_scaled = self.scaler.fit_transform(X_treino)
        self.colunas_hmm = colunas_hmm

        matriz_persistencia = self.matriz_persistencia()
        self.escolha_matriz(matriz_transicao, matriz_persistencia)


        #bloco antigo, fazia o ranking somente pelo ibovespa
        #volatilidades = [self.modelo.means_[i][1] for i in range(len(self.estados_possiveis))]
        #ordem_risco = np.argsort(volatilidades)
        #self.mapa_risco = {estado_hmm: nivel_risco for nivel_risco, estado_hmm in enumerate(ordem_risco)}

        #self.transmat_original = self.modelo.transmat_
        #self.transition_probs = {
        #    (i, j): self.transmat_original[np.where(ordem_risco == i)[0][0]][np.where(ordem_risco == j)[0][0]]
        #    for i in self.estados_possiveis for j in self.estados_possiveis
        #}            

        # mapea e ranqueia os estados latentes do hmm pelo score de risco multivariado
        self.mapa_risco = self.classificar_risco_regime()

        self.transition_probs, self.transmat_ordenada = self.construir_matriz_transicao_ordenada(
            matriz_transicao=matriz_transicao,
            matriz_persistencia=matriz_persistencia
        )

        estado_treino = [self.mapa_risco[s] for s in self.modelo.predict(self.X_treino_scaled)]
        df_treino_acoes = pd.DataFrame(df_treino_acoes.assign(Estado_Regime=estado_treino))

        self.carteiras, nomes_carteiras = criar_carteiras_por_regime(
            df_treino_acoes, self.estados_possiveis, ativos_vivos, ativos_risco, 
            colunas_operacao, tempo_regime, metrica_otimizacao, limite_max_por_ativo,
            limite_min_por_ativo, self.metricas_regime, numero_ativos
        )

        self.recompensas, self.metricas_recompensa = calcular_metricas_bellman(
            estados_possiveis=self.estados_possiveis, 
            df_treino_acoes=df_treino_acoes, 
            acoes_disponiveis_dinamico=self.carteiras,
            colunas_operacao=colunas_operacao,
            tempo_regime=tempo_regime,
            metrica_otimizacao=metrica_otimizacao,
            metricas_regime = self.metricas_regime
        )

        _, self.politica_otima = belmann_equation(
            self.estados_possiveis,
            nomes_carteiras,
            self.transition_probs,
            self.recompensas
        )
    def escolha_matriz(self, matriz_transicao, matriz_persistencia):
        if matriz_transicao == "fixa":
            self.modelo.transmat_ = matriz_persistencia.copy()
            self.modelo.params = "smc"

            self.modelo.init_params = "smc" #inicializacao
            self.modelo.fit(self.X_treino_scaled)
            self.transmat_orignal = matriz_persistencia.copy()
        elif matriz_transicao == "adaptativa":
            self.modelo.init_params = "stmc" #t
            self.modelo.params = "stmc" #t

            self.modelo.fit(self.X_treino_scaled)

            self.transmat_orignal = (
                self.modelo.transmat_.copy()
            )
        else:
            raise ValueError(
                "matriz_transicao deve ser "
                "'fixa' ou 'adaptativa'"
            )

    def matriz_persistencia(self):
        return np.array([
            [0.85, 0.12, 0.02, 0.01],
            [0.08, 0.77, 0.12, 0.03],
            [0.02, 0.10, 0.78, 0.10],
            [0.01, 0.02, 0.12, 0.85]
        ])
    
    def construir_matriz_transicao_ordenada(self, matriz_transicao: str, matriz_persistencia: np.ndarray) -> tuple[dict, np.ndarray]:
        """
        Gera as probabilidades de transição ordenadas consistentemente pelo nível de risco (0 a N-1).
        
        Retorna:
            - transition_probs: dict no formato {(i, j): prob} para consumo na equação de Bellman.
            - transmat_ordenada: np.ndarray (N, N) com linhas/colunas alinhadas à escala de risco.
        """
        n_states = len(self.estados_possiveis)
        
        if matriz_transicao == "fixa":
            # A matriz fixa já é canônica (0: Bull ... N-1: Pânico/Estresse)
            transmat_ordenada = matriz_persistencia.copy()
            transition_probs = {
                (i, j): float(transmat_ordenada[i, j])
                for i in self.estados_possiveis
                for j in self.estados_possiveis
            }
            return transition_probs, transmat_ordenada

        elif matriz_transicao == "adaptativa":
            # No modo adaptativo, permuta as probabilidades brutas aprendidas pelo HMM
            estados_por_risco = {risco: estado_hmm for estado_hmm, risco in self.mapa_risco.items()}
            
            transmat_ordenada = np.zeros((n_states, n_states))
            transition_probs = {}
            
            for i in self.estados_possiveis:
                for j in self.estados_possiveis:
                    raw_i = estados_por_risco[i]
                    raw_j = estados_por_risco[j]
                    prob = float(self.transmat_orignal[raw_i, raw_j])
                    
                    transition_probs[(i, j)] = prob
                    transmat_ordenada[i, j] = prob
                    
            return transition_probs, transmat_ordenada

        else:
            raise ValueError(f"Tipo de matriz '{matriz_transicao}' não reconhecido. Use 'fixa' ou 'adaptativa'.")

    def classificar_risco_regime(self):

        estados_preditos = self.modelo.predict(self.X_treino_scaled)

        df = self.dados_treino_original.copy()

        df['estado'] = estados_preditos

        estatisticas = df.groupby("estado").agg({
            #"ibovespa_br_returns": "mean",
            "vix_zscore": "mean",
            "risco_brasil_zscore": "mean",
            "dolar_cambio_livre_p_tax_zscore": "mean",
            "expectativa_selic_1y_zscore": "mean"
        })


        score = (
            #- 0.30 * estatisticas["ibovespa_br_returns"] +
            0.30 * estatisticas["vix_zscore"] +
            0.30 * estatisticas["risco_brasil_zscore"] +
            0.25 * estatisticas['expectativa_selic_1y_zscore'] +
            0.15 * estatisticas['dolar_cambio_livre_p_tax_zscore'] 
            )

        ranking = score.rank(method='first').astype(int) - 1

        estatisticas['score_risco'] = score
        estatisticas['nivel_risco'] = ranking

        self.relatorios_regimes = estatisticas
        return ranking.to_dict()

    def criar_regimes(self):
        return {
            0: {
                "nome": "Bull_Baixa_Vol",
                "descricao": "Ambiente favorável, baixa volatilidade"
            },
            1: {
                "nome": "Transicao_Normal",
                "descricao": "Mercado normal ou transição"
            },
            2: {
                "nome": "Correcao",
                "descricao": "Ambiente defensivo, risco elevado"
            },
            3: {
                "nome": "Panico_Alta_Vol",
                "descricao": "Estresse extremo, preservação de capital (CVaR)"
            }
        }