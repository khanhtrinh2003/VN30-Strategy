import pandas as pd
import numpy as np

def income(sector="Ngân hàng", field="Chi phí dự phòng rủi ro tín dụng"):
    df = pd.read_excel("D:\KTrinh\python\VN30-Strategy\Fundamental\income.xlsx").ffill()
    df=df.set_index(df.columns[0])
    df = df.groupby([df.index,"CHỈ TIÊU","ticker"]).agg(sum).T[sector]

    close = pd.read_csv("D:\KTrinh\python\VN30-Strategy\Data\close.csv", index_col="TradingDate", parse_dates=True)
    df1 = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    df1.index = df1.index.to_period('Q').strftime('Q%q %Y')
    df=df[field]
    df1.loc[df.index,df.columns]=df.loc[df.index,df.columns]

    df1.ffill(inplace=True)
    df1.fillna(0,inplace=True)
    df1.index=close.index
        
    return df1