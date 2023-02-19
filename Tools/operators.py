import pandas as pd
import numpy as np
import statsmodels.api as sm

# Arithmetic Operators
def ceiling(x):
    return np.ceil(x)

def floor(x):
    return np.floor(x)

def exp(x):
    return np.exp(x)

def log(x):
    return np.log(x)

def round(x):
    return np.round(x)

def sign(x):
    return np.sign(x)

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

def clamp(x, lower = 0, upper = 0, inverse = False, mask = ""):
    q = if_else(x < lower, lower, x)
    u = if_else(q > upper, upper, q)
    v = if_else(x > lower & x < upper, mask, x)
    return if_else(inverse, v, u)

def left_tail(x, maximum = 0):
    return if_else(x>maximum,np.nan,x)

def right_tail(x, minimum = 0):
    return if_else(x<minimum,np.nan,x)

def tail(x, lower = 0, upper = 0, newval = 0):
    return if_else(lower<x & x<upper,newval,x)

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

def truncate_values(x, max_percent=0.1):
    """
    Truncate all values of x to max_percent of the sum of all values.
    
    Parameters:
    x (pandas.DataFrame): input DataFrame with instrument values
    max_percent (float): maximum percentage (in decimal notation) that each value can have
    
    Returns:
    pandas.DataFrame: output DataFrame with truncated values
    """
    # Compute sum of all values
    total = x.sum(axis=1)
    
    # Compute maximum allowed value for each row
    max_value = total * max_percent
    
    # Truncate values that exceed the maximum
    return x.apply(lambda row: np.minimum(row, max_value[row.name]), axis=1)
   
def zscore(x):
    return x.sub(x.mean(axis=1),axis=0).div(x.std(axis=1),axis=0)

# Group Operators
def group_count(x, group):
    df = pd.DataFrame()
    g = group.values
    v = x.values
    va = []
    for i in range(len(g)):
        df.index = g[i]
        df["v"] = v[i]
        df.v=df.groupby(df.index).agg(len)
        va.append(df["v"].to_list())
    df = pd.DataFrame(va)
    df.index = x.index
    df.columns = x.columns
    return df

def group_max(x, group):
    df = pd.DataFrame()
    g = group.values
    v = x.values
    va = []
    for i in range(len(g)):
        df.index = g[i]
        df["v"] = v[i]
        df.v=df.groupby(df.index).agg(np.max)
        va.append(df["v"].to_list())
    df = pd.DataFrame(va)
    df.index = x.index
    df.columns = x.columns
    return df

def group_means(x,group):
    df = pd.DataFrame()
    g = group.values
    v = x.values
    va = []
    for i in range(len(g)):
        df.index = g[i]
        df["v"] = v[i]
        df.v=df.groupby(df.index).agg(np.mean)
        va.append(df["v"].to_list())
    df = pd.DataFrame(va)
    df.index = x.index
    df.columns = x.columns
    return df

def group_median(x, group):
    df = pd.DataFrame()
    g = group.values
    v = x.values
    va = []
    for i in range(len(g)):
        df.index = g[i]
        df["v"] = v[i]
        df.v=df.groupby(df.index).agg(np.median)
        va.append(df["v"].to_list())
    df = pd.DataFrame(va)
    df.index = x.index
    df.columns = x.columns
    return df

def group_min(x, group):
    df = pd.DataFrame()
    g = group.values
    v = x.values
    va = []
    for i in range(len(g)):
        df.index = g[i]
        df["v"] = v[i]
        df.v=df.groupby(df.index).agg(np.min)
        va.append(df["v"].to_list())
    df = pd.DataFrame(va)
    df.index = x.index
    df.columns = x.columns
    return df

def group_neutralize(x, group):
    return x - group_means(x,group)

def group_normalize(x, group, scale=1):
    return x*scale/group_sum(abs(x),group)

def group_percentage(x, group, percentage=0.5):
    pass

def group_rank(x, group):
    df = pd.DataFrame()
    g = group.values
    v = x.values
    va = []
    for i in range(len(g)):
        df.index = g[i]
        df["v"] = v[i]
        df.v=df.groupby(df.index).rank()
        va.append(df["v"].to_list())
    df = pd.DataFrame(va)
    df.index = x.index
    df.columns = x.columns
    return df

def group_scale(x, group):
    return (x-group_min(x,group))/(group_max(x,group)-group_min(x,group))

def group_sum(x, group):
    df = pd.DataFrame()
    g = group.values
    v = x.values
    va = []
    for i in range(len(g)):
        df.index = g[i]
        df["v"] = v[i]
        df.v=df.groupby(df.index).agg(np.sum)
        va.append(df["v"].to_list())
    df = pd.DataFrame(va)
    df.index = x.index
    df.columns = x.columns
    return df

def group_std(x, group):
    df = pd.DataFrame()
    g = group.values
    v = x.values
    va = []
    for i in range(len(g)):
        df.index = g[i]
        df["v"] = v[i]
        df.v=df.groupby(df.index).agg(np.std)
        va.append(df["v"].to_list())
    df = pd.DataFrame(va)
    df.index = x.index
    df.columns = x.columns
    return df    

def group_zscore(x, group):
    return group_neutralize(x,group)/group_std(x,group)

# Time Series Operators
def days_from_last_change(x):
    pass

def ts_weighted_delay(x, k=0.5):
    """Instead of replacing today’s value with yesterday’s as in ts_delay(x,1), it assigns weighted average of today’s and yesterday’s values with weight on today’s value being k and yesterday’s being (1-k)"""
    return k*x + (1-k)*ts_delay(x,1)

def hump(x, hump = 0.01):
    pass

def hump_decay(x, p=0):
    return if_else(abs(x - ts_delay(x))> p * abs(x + ts_delay(x)), x, ts_delay(x))

def inst_tvr(x, d):
    pass

def jump_decay(x, d, sensitivity=0.5, force=0.1):
    return if_else(abs(x-ts_delay(x, 1)) > sensitivity * ts_std(x,d), ts_delay(x,1) + ts_delta(x, 1) * force, x)

def kth_element(x, d, k=1):
    pass

def last_diff_value(x, d):
    pass

def ts_arg_max(x, d):
    return x.rolling(d).agg(np.argmax)

def ts_arg_min(x, d):
    return x.rolling(d).agg(np.argmin)

def ts_av_diff(x, d):
    return x-ts_mean(x,d)

def ts_backfill(x,lookback = 2, k=1, ignore="NAN"):
    pass

def ts_co_kurtosis(y, x, d):
    pass

def ts_corr(x, y, d):
    
    correl = pd.DataFrame()
    corr1 = pd.DataFrame()
    # Iterate over each column in df
    df = pd.DataFrame()
    for col1 in y.columns:
            # Calculate the Correlation between the two columns and store the result in the covariance dataframe
            df['a'] = y[col1]
            df['b'] = x[col1]
            
            col_name = f'{col1}'
            correl[col_name] = df.rolling(d).corr().unstack()['a']['b']
            corr1 = pd.concat([corr1,correl])
    return corr1

def ts_co_skewness(y, x, d):
    pass

def ts_count_nans(x ,d):
    pass

def ts_covariance(y, x, d):
    # Create an empty dataframe to store the covariance results
    covariance = pd.DataFrame()
    cov1 = pd.DataFrame()
    # Iterate over each column in df
    df = pd.DataFrame()
    for col1 in y.columns:
            # Calculate the covariance between the two columns and store the result in the covariance dataframe
            df['a'] = y[col1]
            df['b'] = x[col1]
            
            col_name = f'{col1}'
            covariance[col_name] = df.rolling(d).cov().unstack()['a']['b']
            cov1 = pd.concat([cov1,covariance])
    return cov1


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
    """Returns x - ts_min(x, d)"""
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
    return x.rolling(10).agg(lambda i: np.quantile(i, q=percentage))

def ts_poly_regression(x, y, d,k=1):
    """
    Calculates y - Ey where Ey = x + x^2 + ... + x^k over d days
    
    Parameters:
    x (pandas.DataFrame): input data
    y (pandas.DataFrame): data to subtract mean from
    k (int): highest power of x to include in the mean calculation
    d (int): number of days to include in the mean calculation
    
    Returns:
    pandas.DataFrame: y - Ey
    """
    # Calculate the rolling mean of x^i for i in [1, k]
    ex = sum([x**i for i in range(1, k+1)])
    ex = ex.rolling(d).mean()
    
    # Subtract rolling mean from y
    result = y - ex
    
    return result


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
    """
    Z-score is a numerical measurement that describes a value's relationship to the mean of a group of values. Z-score is measured in terms of standard deviations from the mean: (x - tsmean(x,d)) / tsstddev(x,d)
    """
    return((x-ts_mean(x,d))/ts_std(x,d))

def ts_entropy(x,d):
    pass



    




