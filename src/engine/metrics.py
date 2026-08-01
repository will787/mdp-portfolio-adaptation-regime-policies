import numpy as np 


def cagr(returns):
    dias = len(returns)
    anos = dias / 252
    
    retorno_total = np.prod(1 + returns)
    cagr = retorno_total**(1 / anos) - 1
    
    return cagr

def vol(returns):
    return np.std(returns) * np.sqrt(252)

def sharpe(returns, rf):

    rf = (1 + rf.mean()) ** 252 - 1
    sharpe = (cagr(returns) - rf) / (vol(returns) + 1e-9)

    return sharpe

def sortino(returns, rf):
    rf = (1 + rf.mean()) ** 252 - 1
    ret_neg = returns[returns < 0]

    if len(ret_neg):
        downside = np.sqrt(np.mean(ret_neg**2)) * np.sqrt(252)
        sortino = (cagr(returns) - rf) / downside
    else:
        sortino = np.nan
    
    return sortino
    

def max_drawdown(returns):

    equity = np.cumprod(1 + returns)

    peak = np.maximum.accumulate(equity)

    drawdown = (equity - peak) / peak

    return drawdown.min()

def calmar(returns):

    dd = abs(max_drawdown(returns))

    if dd == 0:
        return np.nan

    return cagr(returns) / dd
