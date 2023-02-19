import pandas as pd
import numpy as np
from vnstock import *


def fndl(field="asset"):
    data = pd.read_csv("D:/KTrinh/python/VN30-Strategy/Fundamental/total.csv",index_col="CHỈ TIÊU")
    # data.drop('Unnamed: 0',axis=1, inplace=True)
    df = data.groupby([data.index,"ticker"]).agg(sum).T
    
    close = pd.read_csv("D:\KTrinh\python\VN30-Strategy\Data\close.csv", index_col="TradingDate", parse_dates=True)
    df1 = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    df1.index = df1.index.to_period('Q').strftime('Q%q %Y')
    df=df[field]
    df1.loc[df.index,df.columns]=df.loc[df.index,df.columns]

    df1.ffill(inplace=True)
    df1.fillna(0,inplace=True)
    df1.index=close.index    
    return df1








# def income(industry="Ngân hàng", field="Chi phí dự phòng rủi ro tín dụng"):
#     df = pd.read_excel("D:\KTrinh\python\VN30-Strategy\Fundamental\income.xlsx").ffill()
#     df=df.set_index(df.columns[0])
#     df = df.groupby([df.index,"CHỈ TIÊU","ticker"]).agg(sum).T[industry]

#     close = pd.read_csv("D:\KTrinh\python\VN30-Strategy\Data\close.csv", index_col="TradingDate", parse_dates=True)
#     df1 = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
#     df1.index = df1.index.to_period('Q').strftime('Q%q %Y')
#     df=df[field]
#     df1.loc[df.index,df.columns]=df.loc[df.index,df.columns]

#     df1.ffill(inplace=True)
#     df1.fillna(0,inplace=True)
#     df1.index=close.index
        
#     return df1

# def balance(industry="Ngân hàng", field="TỔNG TÀI SẢN"):
#     df = pd.read_excel("D:/KTrinh/python/VN30-Strategy/Fundamental/balancesheet.xlsx").ffill()
#     df=df.set_index(df.columns[0])
#     df = df.groupby([df.index,"CHỈ TIÊU","ticker"]).agg(sum).T[industry]

#     close = pd.read_csv("D:\KTrinh\python\VN30-Strategy\Data\close.csv", index_col="TradingDate", parse_dates=True)
#     df1 = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
#     df1.index = df1.index.to_period('Q').strftime('Q%q %Y')
#     df=df[field]
#     df1.loc[df.index,df.columns]=df.loc[df.index,df.columns]

#     df1.ffill(inplace=True)
#     df1.fillna(0,inplace=True)
#     df1.index=close.index
        
#     return df1

# def cashflow(industry="Ngân hàng", field="Lưu chuyển tiền thuần từ các hoạt động sản xuất kinh doanh"):
#     df = pd.read_excel("D:/KTrinh/python/VN30-Strategy/Fundamental/cashflow.xlsx").ffill()
#     df=df.set_index(df.columns[0])
#     df = df.groupby([df.index,"CHỈ TIÊU","ticker"]).agg(sum).T[industry]

#     close = pd.read_csv("D:\KTrinh\python\VN30-Strategy\Data\close.csv", index_col="TradingDate", parse_dates=True)
#     df1 = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
#     df1.index = df1.index.to_period('Q').strftime('Q%q %Y')
#     df=df[field]
#     df1.loc[df.index,df.columns]=df.loc[df.index,df.columns]

#     df1.ffill(inplace=True)
#     df1.fillna(0,inplace=True)
#     df1.index=close.index
        
#     return df1

# def get_balance_field(industry="Ngân hàng"):
#     bl = pd.read_excel("D:/KTrinh/python/VN30-Strategy/Fundamental/field/bl_field.xlsx")
#     return bl[industry].dropna().tolist()

# def get_balance_field(industry="Ngân hàng"):
#     inc = pd.read_excel("D:/KTrinh/python/VN30-Strategy/Fundamental/field/inc_field.xlsx")
#     return inc[industry].dropna().tolist()

# def get_balance_field(industry="Ngân hàng"):
#     cf = pd.read_excel("D:/KTrinh/python/VN30-Strategy/Fundamental/field/cf_field.xlsx")
#     return cf[industry].dropna().tolist()

# def get_industry():
#     industry = []
#     ticket = ['ACB', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'MBB', 'MSN', 'MWG', 'NVL', 'PDR', 'PLX', 'POW', 'SAB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE']
#     for i in ticket:
#         try:
#             a = ticker_overview(i)["industry"][0]
#             if a not in industry:
#                 industry.append(a)
#         except Exception:
#             continue
#     return industry        
