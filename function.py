import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
pd.options.plotting.backend = "plotly"
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

    def get_pnl(self):
        i=np.sum(self.weights*self.returns.shift(-1),axis=1)+1
        a = np.cumprod(i)
        return a

    def get_return(self):
        i=np.sum(self.weights*self.returns.shift(-1),axis=1)+1
        a = np.prod(i)**(252/len(i))-1
        return a

    def get_sharpe(self):
        try:
            return self.get_return()/(np.std(np.sum(self.weights*self.returns.shift(-1),axis=1)+1)*np.sqrt(252))
        except Exception:
            return 0
        
    def get_turnover(self):
        
        weights_t = self.weights.values[1:,:]
        weights_t1 = self.weights.values[:-1,:]
        turnover = np.nansum(np.abs(weights_t - weights_t1), axis = 1)
        return np.mean(turnover)          

    def get_summary(self):
        return pd.DataFrame({'Return': [self.get_return()], 
                             'Sharpe': [self.get_sharpe()], 
                             'Turnover': [self.get_turnover()]})
        
    def plot_pnl(self):
        plt.plot(self.get_pnl())
        plt.ylabel("Cummulative return")
        plt.xlabel("Date")
        plt.title("PnL")

