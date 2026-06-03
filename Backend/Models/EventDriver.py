"""
MODEL # Exemplo ações b3 com balanço
filtra ações minimamente liquidas e com lançamento de balanço (mercado fechado e aberto) 
precifica surpresa (actual - forecast), se >< percentile da volatilidade então C/V

Exemplo resumido (ação XYZ, pré-balanço)
Dados:
Preço atual = R$ 50.00 -> Forecast de lucro -> R$ 1,00/ação
Realizado = R$ 1,20 → surpresa = +0,20 (20% acima)
Volatilidade histórica diária = 2%
Percentil 90 da vol = 3%
GARCH prevê vol de 2,5% para amanhã.

Regra: Se surpresa (20%) > percentil 90 (3%) → sinal ativo.

Execução (duas opções): Entra no momento do lançamento ou dependendo da abertura com opções ou a vista

STRAT
surpresa = real_value - forecast



"""


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ==================================================
# 1. Cálculo de β_0 (sensibilidade histórica)
# ==================================================
def calcular_beta_0(historico_surpresas: np.ndarray, historico_retornos: np.ndarray) -> float:
    """
    β_0 = coeficiente angular de uma regressão linear:
    retorno_t = β_0 * surpresa_t + erro

    Parâmetros:
        historico_surpresas: array com surpresas passadas (z_t)
        historico_retornos: array com retornos observados do ativo (Δ%_t)
    
    Retorna:
        β_0 estimado
    """
    model = LinearRegression()
    # Reshape para 2D (amostras, 1 feature)
    model.fit(historico_surpresas.reshape(-1, 1), historico_retornos)
    return float(model.coef_[0])

# ==================================================
# 2. Cálculo de m_{r_t} (fator de ciclo de mercado)
# ==================================================
def calcular_m_rt(volatilidade_atual: float,
                  volatilidade_historica_media: float,
                  regime: str = 'normal') -> float:
    """
    m_{r_t} quantifica como o ciclo de mercado amplifica/reduz o impacto da surpresa.
    Exemplo simples baseado em volatilidade relativa e regimes.

    Parâmetros:
        volatilidade_atual: ex: desvio padrão dos retornos diários dos últimos 20 dias
        volatilidade_historica_media: média de longo prazo da volatilidade
        regime: 'normal', 'alta_vol', 'baixa_vol' (ou pode ser inferido automaticamente)
    
    Retorna:
        m_rt > 1 amplifica o movimento, < 1 atenua.
    """
    razao_vol = volatilidade_atual / volatilidade_historica_media
    
    if regime == 'alta_vol':
        return razao_vol * 1.2   # ciclo de estresse amplifica ainda mais
    elif regime == 'baixa_vol':
        return max(0.5, razao_vol * 0.8)
    else:  # normal
        return razao_vol

# Versão automática que detecta o regime baseado em percentis históricos
def detectar_regime(volatilidade_atual: float,
                    serie_historica_vol: np.ndarray) -> str:
    """Retorna 'baixa_vol', 'normal' ou 'alta_vol' baseado nos percentis 30 e 70."""
    p30 = np.percentile(serie_historica_vol, 30)
    p70 = np.percentile(serie_historica_vol, 70)
    if volatilidade_atual < p30:
        return 'baixa_vol'
    elif volatilidade_atual > p70:
        return 'alta_vol'
    else:
        return 'normal'

# ==================================================
# 3. Cálculo de λ_t (fator dinâmico próprio)
# ==================================================
def calcular_lambda_t(momentum: float,
                      sentimento: float = 1.0,
                      alavancagem: float = 1.0) -> float:
    """
    λ_t incorpora efeitos como tendência de curto prazo, sentimento do mercado, 
    ou apetite por risco.

    Exemplo:
        λ_t = momentum_clipped * sentimento * fator_alavancagem

    Parâmetros:
        momentum: retorno acumulado nos últimos N dias (ex: 0.03 para +3%)
        sentimento: proxy, ex: 0.8 (negativo) a 1.2 (positivo)
        alavancagem: 1.0 sem alavancagem, >1 se posições alavancadas
    
    Retorna:
        λ_t (tipicamente entre 0.5 e 2.0)
    """
    # Limita momentum para evitar valores extremos
    momentum_efetivo = np.clip(1 + momentum, 0.7, 1.5)
    return momentum_efetivo * sentimento * alavancagem

# ==================================================
# 4. Função principal que integra tudo
# ==================================================
def precificar_surpresa(z_t: float, beta_0: float, m_rt: float, lambda_t: float) -> float:
    """
    Calcula a magnitude do movimento do ativo (Δ%_t) dado o tamanho da surpresa
    e os parâmetros do ciclo de mercado.

    Modelo: Δ%_t = z_t * β_0 * m_{r_t} * λ_t

    Parâmetros:
    -----------
    z_t : float
        Tamanho da surpresa no tempo t (ex: diferença entre esperado e realizado).
    beta_0 : float
        Coeficiente base de sensibilidade do ativo.
    m_rt : float
        Fator de mercado no ciclo corrente (ex: volatilidade, liquidez).
    lambda_t : float
        Fator de ajuste dinâmico (ex: alavancagem, sentimento).

    Retorna:
    --------
    float
        Valor de Δ%_t (em unidades absolutas, ex: 0.05 para 5%).
    """
    return z_t * beta_0 * m_rt * lambda_t

# Exemplo de uso:
# Suponha uma surpresa de 0.02 (2%), beta_0=0.8, m_rt=1.2, lambda_t=1.0
# movimento = precificar_surpresa(0.02, 0.8, 1.2, 1.0)

# ==================================================
# EXEMPLO COMPLETO DE USO
# ==================================================
if __name__ == "__main__":
    # 1. Dados históricos para estimar β_0
    # (substitua pelos seus dados reais)
    surpresas_historicas = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
    retornos_historicos  = np.array([0.008, -0.004, 0.018, -0.009, 0.012])
    beta_0 = calcular_beta_0(surpresas_historicas, retornos_historicos)
    print(f"β_0 estimado: {beta_0:.4f}")

    # 2. Cálculo de m_rt com dados de mercado atuais
    vol_atual = 0.25   # volatilidade anualizada 25%
    vol_historica_media = 0.20
    serie_vol_historica = np.array([0.18, 0.22, 0.19, 0.27, 0.24, 0.21])  # exemplo
    regime = detectar_regime(vol_atual, serie_vol_historica)
    m_rt = calcular_m_rt(vol_atual, vol_historica_media, regime)
    print(f"m_rt = {m_rt:.3f} (regime {regime})")

    # 3. Cálculo de λ_t (seu modelo pessoal)
    momentum_20d = 0.04   # +4% nos últimos 20 dias
    sentimento_mercado = 1.1  # otimista
    lambda_t = calcular_lambda_t(momentum_20d, sentimento_mercado)
    print(f"λ_t = {lambda_t:.3f}")

    # 4. Surpresa do momento (ex: IPO da SpaceX)
    dado_atual = 95.0   # preço de abertura real (simulado)
    forecast = 100.0    # sua previsão ou consenso
    z_t = (dado_atual - forecast) / forecast # surpresa = -5 (negativa)
    print(f"Surpresa z_t = {z_t:.2f}")

    # 5. Movimento esperado
    movimento = precificar_surpresa(z_t, beta_0, m_rt, lambda_t)
    print(f"Δ%_t esperado = {movimento:.4f} ({movimento*100:.2f}%)")


# NOTE Na prática
# Se o preço atual é R$ 95 e a surpresa é -5%, o modelo espera que o preço vá para:
# preço_alvo = 95 * (1 + Δ%_t) = 95 * (1 - 0,0743) ≈ R$ 87,94

    # percentil_90 = np.percentile(historico_delta, 90)
    # percentil_10 = np.percentile(historico_delta, 10)

    # if Δ%_t > percentil_90:
    #     sinal = "LONG"
    # elif Δ%_t < percentil_10:
    #     sinal = "SHORT"







