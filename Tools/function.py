import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from vnstock import *

close = pd.read_csv("D:\KTrinh\python\VN30-Strategy\Data\close.csv", index_col="TradingDate", parse_dates=True)
returns = close.pct_change()

def weights(alpha,neutrali=0):
    # Normalize
    alpha = alpha.div(alpha.abs().sum(axis=1), axis=0)
    # Set none if nan > 20
    di = alpha.index.where(alpha.isnull().sum(axis=1) >= 20)
    di = di[~np.isnan(di)]
    alpha.loc[di] = None
    if neutrali == 1:
        alpha = alpha.sub(alpha.mean(axis=1),axis=0)
        positive_alpha = alpha[alpha > 0]
        negative_alpha = alpha[alpha < 0]
        positive_sum = positive_alpha.sum(axis=1)
        negative_sum = negative_alpha.abs().sum(axis=1)
        positive_alpha = positive_alpha.div(positive_sum, axis=0)
        negative_alpha = negative_alpha.div(negative_sum, axis=0)
        alpha[alpha > 0] = positive_alpha
        alpha[alpha < 0] = negative_alpha

    # Max stock weight = 0.2
    alpha[alpha > 0.2] = 0.2
    return alpha

def prob_weights(prices,lag):    
    lag = lag
    diff_data = prices.diff()
    abs_diff_data = abs(diff_data)
    nominator_data =(abs_diff_data+diff_data)/2
    a = (abs_diff_data-diff_data)/2
    prob = nominator_data.rolling(lag).sum()/((abs_diff_data).rolling(lag).sum())
    prob[prob<=0.5]=-prob[prob<=0.5]
    return prob

def marko_weights(prices, lag):
    def weight_mar(prices):
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

    mark = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for i in range(int(len(prices)/lag)):
        mark.iloc[(i+1)*lag,:] = weight_mar(prices[i*lag:(i+1)*lag])
    mark.ffill(inplace=True)      
    
    return mark

def plot_vnindex():
    m=get_index_series(index_code='VNINDEX', time_range='TenYears')
    m["i"]=pd.to_datetime(m["tradingDate"])
    plt.plot(m["i"],np.cumsum(m["indexValue"].pct_change()), label="VNINDEX")
    plt.legend()

def save_weights(weight,x):
    d = pd.read_csv("Weights\Weights.csv",index_col="ticker").drop(columns="Delta")
    d1 = pd.DataFrame(weight.iloc[-1,:].sort_values(ascending=False))
    d1.columns = [x]
    d = pd.merge(d,d1, left_index=True,right_index=True)
    d["Delta"] = d.iloc[:,len(d.columns)-1]-d.iloc[:,len(d.columns)-2]
    d = d.sort_values(by="Delta",ascending=False)
    d.to_csv("Weights\Weights.csv")

class Simresult():
    def __init__(self,weights,returns):
        self.weights = weights
        self.returns = returns
        self.ret = np.sum(self.weights*self.returns.shift(-1),axis=1)

    def get_pnl(self):
        return np.cumsum(self.ret)

    def get_return(self):
        i = self.ret
        return i.groupby(i.index.year).agg(np.mean)*252

    def get_sharpe(self):
        i = self.ret
        return self.get_return()/(i.groupby(i.index.year).agg(np.std)*np.sqrt(252))
        
    def get_turnover(self):
        weights_t = self.weights
        weights_t1 = self.weights.shift(1)
        turnover = np.sum(np.abs(weights_t - weights_t1),axis=1).groupby(self.weights.index.year).agg(np.mean)
        return turnover    

    def get_fitness(self):
        turnover = self.get_turnover()
        turnover = np.maximum(turnover, 0.125)
        fitness = self.get_sharpe() * np.sqrt(np.abs(self.get_return() / turnover))
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
        f = s*np.sqrt(np.abs(r/np.maximum(t, 0.125)))
        m = r/t*1000
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

class Stimulate():
    def __init__(self,alpha):
        self.alpha = alpha
        self.non_neu = weights(self.alpha,neutrali=0)
        self.neu = weights(self.alpha,neutrali=1)
        self.ketqua_non = Simresult(self.non_neu,returns)
        self.ketqua_neu = Simresult(self.neu,returns)

    def overall(self):
        print("Overall of non neutralize")
        print(self.ketqua_non.get_overall())
        print("Overall of neutralize")
        print(self.ketqua_neu.get_overall())

    def summary(self):
        print("Summary of non neutralize")
        print(self.ketqua_non.get_summary())
        print("Summary of neutralize")
        print(self.ketqua_neu.get_summary())

    def plot_pnl(self):
        self.ketqua_neu.plot_pnl("Neutralizing")
        self.ketqua_non.plot_pnl("Non neutralizing")
        plot_vnindex()

    def get_weights(self,non_neu=0):
        if non_neu==0:
            print(self.neu.iloc[-1,:].sort_values(ascending=False))
        else:
            print("Neutralization")
            print(self.neu.iloc[-1,:].sort_values(ascending=False))
            print("Non-Neutralization")
            print(self.non_neu.iloc[-1,:].sort_values(ascending=False))
