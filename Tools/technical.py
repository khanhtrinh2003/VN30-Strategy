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
        """Moving Average Envelop"""
        s = ts_mean(close,20)
        le1 = s * 0.95
        le2 = s * 0.9
        le3 = s * 0.85
        le4 = s * 0.8
        le5 = s * 0.75
        le6 = s * 0.70    

        a1=0.1
        a2=0.2
        a3=0.25
        a4=0.32
        a5=0.4
        a6=0.5

        df = if_else(low < le6,alpha + a6*abs(alpha),
                        if_else(low < le5,alpha + a5*abs(alpha),
                                if_else(low < le4,alpha + a4*abs(alpha),
                                        if_else(low < le3,alpha + a3*abs(alpha),
                                                if_else(low < le2,alpha + a2*abs(alpha),
                                                        if_else(low < le1,alpha+a1*abs(alpha),alpha)
                                                        )
                                                )
                                        )
                                )
                        )
        return df

def rsi(alpha):
        """Relative Strength Index"""
        RSI = (100 - 100 / (1 + ts_sum(if_else(ts_delta(close,1) > 0, ts_delta(close,1), 0), 14) / ts_sum(if_else(ts_delta(close,1)< 0, - ts_delta(close,1),0), 14)))
        SRSI = (RSI-ts_min(RSI,14))/(ts_max(RSI,14)-ts_min(RSI,14))
        return if_else(SRSI <0.1, alpha+0.1*abs(alpha), alpha)

def kst(alpha):
        """Pring's Know Sure Thing (KST)"""
        RCMA1 = ts_decay_linear(ts_delta(close, 10), 10)
        RCMA2 = ts_decay_linear(ts_delta(close, 15), 10)
        RCMA3 = ts_decay_linear(ts_delta(close, 20), 10)
        RCMA4 = ts_decay_linear(ts_delta(close, 30), 15)
        KST = (RCMA1 * 1) + (RCMA2 * 2) + (RCMA3 * 3) + (RCMA4 * 4)
        SIG = ts_decay_linear(KST, 9)
        return if_else(KST < SIG, alpha + 0.1 * abs(alpha), alpha)

def macd(alpha):
        """MACD (Moving Average Convergence/Divergence Oscillator)"""
        EMA_12=(close-ts_delay((ts_sum(close,12)/12),1))*0.1538+ts_delay((ts_sum(close,12)/12),1)
        EMA_26=(close-ts_delay((ts_sum(close,26)/26),1))*0.0741+ts_delay((ts_sum(close,26)/26),1)
        MACD=EMA_12-EMA_26
        return if_else(MACD<0,alpha+0.1*abs(alpha),alpha)

def macd_his(alpha):
        """MACD Histogram"""
        EMA12=ts_decay_linear(close,12)
        EMA26=ts_decay_linear(close,26)
        MACDLine =  EMA12 - EMA26
        SignalLine=ts_decay_linear(MACDLine,9)
        MACDHistogram = MACDLine - SignalLine
        return if_else(MACDHistogram > 0,alpha + 0.5*abs(alpha),alpha)

def cc(alpha):
        """Coppock Curve"""
        ROC11= (ts_delta(close,11)/ts_delay(close,11))*100
        ROC14 = (ts_delta(close,14)/ts_delay(close,14))*100
        RORO = ROC11 + ROC14
        return if_else(ts_decay_linear(RORO,10)>0, alpha + 0.1*abs(alpha), alpha)

def roc(alpha):
        """Rate of Change (ROC) and Momentum"""
        ROC125 = (ts_delta(close,125)/ts_delay(close,15))*100
        ROC21 = (ts_delta(close,21)/ts_delay(close,21))*100
        return if_else(ROC125 > 0 & ROC21 <-8, alpha +0.1*abs(alpha), alpha)


def std_indicator(alpha):
        """Standard Deviation (Volatility)"""
        return if_else(ts_std(close,20)/(ts_sum(close,20)/20)<0.2, alpha + 0.1*abs(alpha), alpha)

def ui(alpha):
        """Ulcer Index"""
        PD = (close -ts_max(close,14))/ts_max(close,14)
        SA = ts_sum(PD**2,14)/14
        UI = SA**0.5
        return if_else(UI < 10, alpha +0.1*abs(alpha), alpha)

def ic(alpha):
        """Ichimoku Cloud"""
        bl = (ts_max(high, 26) - ts_min(low, 26))/2
        return if_else(close > bl, alpha + 0.1 * abs(alpha), alpha)

def kc(alpha):
        """Keltner Channels"""
        TR = max(high-low,abs(high-ts_delay(close,1)),abs(low-ts_delay(close,1)))
        ATR = ts_sum(TR,10)/10
        EMA = ts_decay_linear(close,20)
        LC = EMA - 2*ATR
        return if_else(low < LC, alpha + 0.1*abs(alpha), alpha)

def pc(alpha):
        """Price Channels"""
        ll = ts_decay_linear(low, 20)
        return if_else((close < ll & ts_delta(close,1) > 0) , alpha + 0.1 * abs(alpha), alpha)

def cci(alpha):
        """CCI"""
        TP=(high+close+low)/3
        CCI= (TP-(ts_sum(TP,20)/20))/(0.015*ts_std(TP,20))
        return if_else(CCI<=100, alpha+0.1*abs(alpha),alpha)

def mfi(alpha):
        """Money Flow Index (MFI)"""
        RMF = (high + low + close)/3 * volume
        MFR = ts_sum(if_else(ts_delta(close,1) > 0 , RMF, 0), 14)/ts_sum(if_else(ts_delta(close,1) < 0 , RMF, 0), 14)
        MFI = 100 - 100/(1 + MFR)
        return if_else(MFI < 30 , alpha + 0.1 * abs(alpha), alpha)

def ppo(alpha):
        """Percentage Price Oscillator (PPO)"""
        PPO = (ts_decay_exp(close, 0.1538, 12)/ts_decay_exp(close, 0.074, 26) - 1) * 100
        SL = ts_decay_exp(PPO , 0.1, 9)
        return if_else(PPO < SL , alpha + 0.1 * abs(alpha), alpha)

def so1(alpha):
        """Stochastic Oscillator"""
        pK = (close-ts_min(low,14))/(ts_max(high,14)-ts_min(low,14))*100
        pD = ts_sum(pK,3)/3
        return if_else(pK<20, alpha + 0.1*abs(alpha),alpha)

def so2(alpha):
        """Stochastic Oscillator 2"""
        pK = (close-ts_min(low,14))/(ts_max(high,14)-ts_min(low,14))*100
        pD = ts_sum(pK,3)/3
        return if_else(pK > 80, alpha - 0.1*abs(alpha), 
                       if_else(pK<20, alpha + 0.1*abs(alpha), alpha)
                       )

def pmo(alpha):
        """DecisionPoint Price Momentum Oscillator (PMO)"""
        ROC = ts_delta(close,1)/ts_delay(close,1)*100
        EMAROC = ts_decay_linear(ROC,35)*10;PMOLine = ts_decay_linear(EMAROC,20)
        PMOSignalLine = ts_decay_linear(PMOLine,10)
        condition = sum(ts_delta(PMOSignalLine,1)>0,3)
        return if_else(condition==3|condition==0, alpha + 0.1*abs(alpha),alpha)

def bi(alpha):
        """%B Indicator"""
        RMF = (high + low + close)/3 * volume
        MFR = sum(if_else(ts_delta(close,1)>0, RMF, 0), 14)/sum(if_else(ts_delta(close,1)<0, RMF, 0), 14)
        MFI = 100 - 100/(1 + MFR)
        return if_else(MFI>80  &  ts_delay(MFI,1)<80, alpha + 0.1*abs(alpha), alpha)

def wr(alpha):
        """Williams %R"""
        R = (ts_max(high,14)-close)/(ts_max(high,14)-ts_min(low,14))*(-100)
        return if_else(R>=50, alpha+0.1*abs(alpha),alpha)

def rising_star(alpha):
        """Rising sar"""
        lastSAR = ts_delay(close,1) - 0.5*ts_std(close,5)
        RisingSAR = lastSAR + 0.02*max(ts_sum(high>ts_max(high,5),20),10)*(ts_max(high,20)-lastSAR)
        return if_else(close< RisingSAR , alpha + 0.1*abs(alpha),alpha)

def atr(alpha):
        """Average True Range (ATR)"""
        TR = max(high-low,abs(high-ts_delay(close,1)),abs(low-ts_delay(close,1)))
        ATR14 = ts_sum(TR,14)/14;SMA20 = ts_sum(close,20)/20
        return if_else((ATR14/SMA20*100<4), alpha+0.1*abs(alpha), alpha)

def emv(alpha): 
        """Ease of Movement (EMV)"""
        DistanceMoved = (high + low)/2-ts_delay((high + low)/2,1)
        BoxRatio = ((volume/100000000)/(high - low))
        EMV = DistanceMoved/BoxRatio
        SMAEMA = ts_sum(EMV,14)/14
        return if_else(SMAEMA>0, alpha + 0.1*abs(alpha),alpha)

def fi(alpha):
        """Force Index"""
        ForceIndex = (close-ts_delay(close,1))*volume
        ForceIndex14 = ts_decay_linear(ForceIndex,14)
        return if_else(ForceIndex14 > 0 , alpha + 0.1*abs(alpha), alpha)

def mi(alpha):
        """Mass Index"""
        DistanceMoved = high - low
        SMA9 = ts_sum(DistanceMoved,9)/9
        SMA9ofSMA9 = ts_sum(SMA9,9)/9
        RatioofSMAs = SMA9/SMA9ofSMA9
        MassIndex = ts_sum(RatioofSMAs,25)
        return if_else(MassIndex < 26.5, alpha + 0.1*abs(alpha),alpha)

def cmf(alpha):
        """Chaikin Money Flow (CMF)"""
        MFM = ((close-low)-(high-close))/(high-low)
        MFV = MFM*volume
        CFM = ts_sum(MFV,20)/ts_sum(volume,20)
        return if_else(CFM > 0, alpha + 0.1*abs(alpha),alpha)
