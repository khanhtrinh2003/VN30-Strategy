import sys
sys.path.append("Tools")

from function import *
from operators import *

import pandas as pd
import numpy as np

close= pd.read_csv("D:/KTrinh/python/WQ/Data/close.csv",index_col="TradingDate")
close.index = pd.to_datetime(close.index)
high= pd.read_csv("D:/KTrinh/python/WQ/Data/high.csv",index_col="TradingDate")
high.index = pd.to_datetime(high.index)
low= pd.read_csv("D:/KTrinh/python/WQ/Data/low.csv",index_col="TradingDate")
low.index = pd.to_datetime(low.index)
open= pd.read_csv("D:/KTrinh/python/WQ/Data/open.csv",index_col="TradingDate")
open.index = pd.to_datetime(open.index)
volume= pd.read_csv("D:/KTrinh/python/WQ/Data/volume.csv",index_col="TradingDate")
volume.index = pd.to_datetime(open.index)

returns = close.pct_change()

def mae(alpha):
        s = ts_mean(close,20)
        le1 = s * 0.95
        le2 = s * 0.9
        le3 = s * 0.85
        le4 = s * 0.8
        le5 = s * 0.75
        le6 = s * 0.70    
        df = if_else(low < le6,alpha + 0.5*abs(alpha),
                        if_else(low < le5,alpha + 0.4*abs(alpha),
                                if_else(low < le4,alpha + 0.32*abs(alpha),
                                        if_else(low < le3,alpha + 0.25*abs(alpha),
                                                if_else(low < le2,alpha + 0.2*abs(alpha),
                                                        if_else(low < le1,alpha+0.1*abs(alpha),alpha)
                                                        )
                                                )
                                        )
                                )
                        )
        return df

def rsi(alpha):
        RSI = (100 - 100 / (1 + ts_sum(if_else(ts_delta(close,1) > 0, ts_delta(close,1), 0), 14) / ts_sum(if_else(ts_delta(close,1)< 0, - ts_delta(close,1),0), 14)))
        SRSI = (RSI-ts_min(RSI,14))/(ts_max(RSI,14)-ts_min(RSI,14))
        return if_else(SRSI <0.1, alpha+0.1*abs(alpha), alpha)

def kst(alpha):
        #Pring's Know Sure Thing (KST)
        RCMA1 = ts_decay_linear(ts_delta(close, 10), 10)
        RCMA2 = ts_decay_linear(ts_delta(close, 15), 10)
        RCMA3 = ts_decay_linear(ts_delta(close, 20), 10)
        RCMA4 = ts_decay_linear(ts_delta(close, 30), 15)
        KST = (RCMA1 * 1) + (RCMA2 * 2) + (RCMA3 * 3) + (RCMA4 * 4)
        SIG = ts_decay_linear(KST, 9)
        return if_else(KST < SIG, alpha + 0.1 * abs(alpha), alpha)

def macd(alpha):
        #MACD (Moving Average Convergence/Divergence Oscillator)
        EMA_12=(close-ts_delay((ts_sum(close,12)/12),1))*0.1538+ts_delay((ts_sum(close,12)/12),1)
        EMA_26=(close-ts_delay((ts_sum(close,26)/26),1))*0.0741+ts_delay((ts_sum(close,26)/26),1)
        MACD=EMA_12-EMA_26
        return if_else(MACD<0,alpha+0.1*abs(alpha),alpha)

def macd_his(alpha):
        #MACD Histogram
        EMA12=ts_decay_linear(close,12)
        EMA26=ts_decay_linear(close,26)
        MACDLine =  EMA12 - EMA26
        SignalLine=ts_decay_linear(MACDLine,9)
        MACDHistogram = MACDLine - SignalLine
        return if_else(MACDHistogram > 0,alpha + 0.5*abs(alpha),alpha)
