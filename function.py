import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def weights(alpha):
    # Set weight > 0
    alpha = alpha.sub(alpha.min(axis=1), axis=0)
    # Normalize
    alpha = alpha.div(alpha.sum(axis=1), axis=0)
    # Set none if nan > 20
    di = alpha.index.where(alpha.isnull().sum(axis = 1) >= 20)
    di = di[~np.isnan(di)]
    alpha.loc[di] = None
    # Max stock weight = 0.2
    alpha[alpha > 0.2] = 0.2
    return alpha

def weights1(alpha,neutrali=0):
    # Normalize
    alpha = alpha.div(alpha.abs().sum(axis=1), axis=0)
    # Set none if nan > 20
    di = alpha.index.where(alpha.isnull().sum(axis = 1) >= 20)
    di = di[~np.isnan(di)]
    alpha.loc[di] = None
    if neutrali == 1:
        alpha = alpha.sub(alpha.mean(axis=1),axis=0)
    # Max stock weight = 0.2
    alpha[alpha > 0.2] = 0.2
    return alpha

def marko_weights(prices):
    returns = prices.pct_change().mean()
    covariance = prices.pct_change().cov()
    risk_aversion = 1
    n = returns.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = [(0, 1) for i in range(n)]
    def neg_sharpe_ratio(weights):
        portfolio_return = np.sum(returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
        sharpe_ratio = -(portfolio_return - 0.02) / portfolio_volatility
        return sharpe_ratio
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    result = minimize(neg_sharpe_ratio, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

class Simresult():
    def __init__(self,weights,returns):
        self.weights = weights
        self.returns = returns
        self.ret = np.sum(self.weights*self.returns.shift(-1),axis=1)

    def get_pnl(self):
        i=np.cumsum(self.ret)
        return i

    def get_return(self):
        return self.ret.groupby(self.ret.index.year).agg(np.mean)*252

    def get_sharpe(self):
        i = self.ret
        return self.get_return()/(i.groupby(i.index.year).agg(np.std)*np.sqrt(252))
        
    def get_turnover(self):
        weights_t = self.weights
        weights_t1 = self.weights.shift(1)
        turnover = np.sum(np.abs(weights_t - weights_t1),axis=1).groupby(self.weights.index.year).agg(np.mean)
        return turnover    

    def get_fitness(self):
        fitness = self.get_sharpe()*np.sqrt(np.abs(self.get_return()/self.get_turnover()))
        return fitness

    def get_margin(self):
        margin = self.get_return()*1000/self.get_turnover()
        return margin
    
    def get_summary(self):
        return pd.DataFrame({'Return': self.get_return().values, 
                             'Sharpe': self.get_sharpe().values, 
                            'Turnover': self.get_turnover().values,
                            "Fitness": self.get_fitness().values,
                            "Margin": self.get_margin().values
                            }, index=self.get_return().index)    
    
    def get_overall(self):
        r = np.mean(self.ret)*252
        s = r/(np.std(self.ret)*np.sqrt(252))
        t = np.mean(self.get_turnover())
        f = s*np.sqrt(np.abs(r/t))
        m = r/t
        return pd.DataFrame({'Return': [r], 
                             'Sharpe': [s], 
                            'Turnover': [t],
                            "Fitness": [f],
                            "Margin": [m],
                            })
        
    def plot_pnl(self,type=""):
        plt.plot(self.get_pnl(), label=type)
        plt.xlabel("Date")
        plt.title("PnL")
        plt.legend()