import pandas as pd
import numpy as np

# Logical Operators
def if_else(condition, x, y):
    df = pd.DataFrame(np.nan, index=condition.index, columns=condition.columns)
    df[df.columns] = np.where(condition,x,y)
    return df

# Transformational Operators
def arc_cos(x):
    return x.transform(np.arccos)

def arc_sin(x):
    return x.transform(np.arcsin)

def arc_tan(x):
    return x.transform(np.arctan)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return x.transform(np.tanh)            

# Cross Sectional Operators
def rank(x):
    p = x.rank(axis=1,ascending=True)
    return p.sub(p.min(axis=1),axis=0).div(p.max(axis=1).sub(p.min(axis=1),axis=0),axis=0)

def zscore(x):
    return x.sub(x.mean(axis=1),axis=0).div(x.std(axis=1),axis=0)

# Group Operators


# Time Series Operators
def days_from_last_change(x):
    pass

def ts_weighted_delay(x, k=0.5):
    pass

def hump(x, hump = 0.01):
    pass

def hump_decay(x, p=0):
    pass

def inst_tvr(x, d):
    pass

def jump_decay(x, d, sensitivity=0.5, force=0.1):
    pass

def kth_element(x, d, k=1):
    pass

def last_diff_value(x, d):
    pass

def ts_arg_max(x, d):
    pass

def ts_arg_min(x, d):
    pass

def ts_av_diff(x, d):
    return x-ts_mean(x,d)

def ts_backfill(x,lookback = 2, k=1, ignore="NAN"):
    pass

def ts_co_kurtosis(y, x, d):
    pass

def ts_corr(x, y, d):
    pass

def ts_co_skewness(y, x, d):
    pass

def ts_count_nans(x ,d):
    pass

def ts_covariance(y, x, d):
    pass

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

def ts_delay(x, d):
    return x.shift(d)

def ts_delta(x,d):
    return x-x.shift(d)

def ts_ir(x, d):
    return ts_mean(x,d)/ts_std(x,d)

def ts_kurtosis(x, d):
    return x.rolling(d).kurt()

def ts_max(x,d):
    return x.rolling(d).max()

def ts_max_diff(x, d):
    return x-ts_max(x,d)

def ts_mean(x, d):
    return x.rolling(d).mean()

def ts_median(x, d):
    return x.rolling(d).median()

def ts_min(x,d):
    return x.rolling(d).min()

def ts_min_diff(x, d):
    return x-ts_min(x,d)

def ts_min_max_cps(x, d, f = 2):
    return ts_min(x, d) + ts_max(x, d) - f * x

def ts_min_max_diff(x, d, f = 0.5):
    return x - f * (ts_min(x, d) + ts_max(x, d))

def ts_moment(x, d, k=0):
    def central_moment(x, **kwargs):
        k = kwargs['k']
        mean = x.mean()
        centered = x - mean
        return (centered**k).mean()
    
    return x.rolling(d).agg(central_moment, k=k)

def ts_partial_corr(x, y, z, d):
    pass

def ts_percentage(x, d, percentage=0.5):
    pass

def ts_poly_regression(y, x, d, k = 1):
    pass

def ts_product(x,d):
    return x.rolling(d).apply(np.prod)

def ts_rank(x,d):
    return x.rolling(d).rank()

def ts_regression(y, x, d, lag = 0, rettype = 0):
    pass

def ts_returns (x, d, mode = 1):
    if mode==1:
        return  (x - ts_delay(x, d))/ts_delay(x, d)
    elif mode==2:
        return (x - ts_delay(x, d))/((x + ts_delay(x, d))/2)

def ts_scale(x, d, constant = 0):
     return (x - ts_min(x, d)) / (ts_max(x, d) - ts_min(x, d)) + constant

def ts_skewness(x, d):
    return x.rolling(d).skew()

def ts_std(x,d):
    return x.rolling(d).std()

def ts_sum(x,d):
    return x.rolling(d).sum()

def ts_theilsen(x, y, d):
    pass

def ts_triple_corr(x, y, z, d):
    pass

def ts_zscore(x,d):
    return((x-x.rolling(d).mean())/x.rolling(d).std())

def ts_entropy(x,d):
    pass



    




