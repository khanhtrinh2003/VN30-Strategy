import pandas as pd
import numpy as np

def rank(x):
    return x.rank(axis=1,ascending=True)

def ts_sum(x,d):
    return x.rolling(d).sum()

def ts_delta(x,d):
    return x-x.shift(d)

def ts_std(x,d):
    return x.rolling(d).std()

def ts_min(x,d):
    return x.rolling(d).min()

def ts_max(x,d):
    return x.rolling(d).max()

def ts_product(x,d):
    return x.rolling(d).apply(np.prod)

def ts_median(x, d):
    return x.rolling(d).median()

def ts_mean(x, d):
    return x.rolling(d).mean()

def ts_delay(x, d):
    return x.shift(d)

def ts_ir(x, d):
    return ts_mean(x,d)/ts_std(x,d)

def ts_rank(x,d):
    return x.rolling(d).rank()

def ts_skewness(x, d):
    return x.rolling(d).skew()

def ts_min_diff(x, d):
    return x-ts_min(x,d)

def ts_max_diff(x, d):
    return x-ts_max(x,d)

def ts_kurtosis(x, d):
    return x.rolling(d).kurt()

def zscore(x):
    return x.sub(x.mean(axis=1),axis=0).div(x.std(axis=1),axis=0)

def ts_zscore(x,d):
    return((x-x.rolling(d).mean())/x.rolling(d).std())
    
def ts_av_diff(x, d):
    return x-ts_mean(x,d)

def ts_moment(x, d, k=0):
    def central_moment(x, **kwargs):
        k = kwargs['k']
        mean = x.mean()
        centered = x - mean
        return (centered**k).mean()
    
    return x.rolling(d).agg(central_moment, k=k)

def ts_scale(x, d, constant = 0):
    return (x - ts_min(x, d)) / (ts_max(x, d) - ts_min(x, d)) + constant


def ts_decay_exp(x,d,f=1):    
    def TS_Decay_Exp_Window(x, d, factor = 1):
        weights = np.power(factor, np.arange(d))
        weighted_sum = np.sum(x[-d:] * weights[::-1])
        denominator = np.sum(weights)
        return weighted_sum / denominator    
    # Assuming x is a pandas DataFrame with a column called 'value'
    rolling_mean = x.rolling(window=d).apply(TS_Decay_Exp_Window, args=(d,f))
    return rolling_mean

def ts_decay_linear(x,d):
    def TS_Decay_Linear(x, d):
        weights = np.arange(1, d+1)[::-1]
        weighted_sum = np.sum(x[-d:] * weights)
        denominator = np.sum(weights)
        return weighted_sum / denominator    
    # Assuming x is a pandas DataFrame with a column called 'value'
    rolling_mean = x.rolling(window=d).apply(TS_Decay_Linear, args=(d,))
    return rolling_mean


