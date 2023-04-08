# alpha 
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.api as sm
rf=0.0
df=pd.read_html('https://www.taifex.com.tw/cht/9/futuresQADetail')
df=df[0]
df=df[['證券名稱','證券名稱.1']]
df=df.iloc[0:100]
start=dt.datetime(2006,1,2)
end=dt.datetime(2022,12,31)
ticker=[str(i)+'.TW' for i in df['證券名稱']]
df_yf=yf.download(ticker,start=start,end=end)
n_close=len(df_yf['Adj Close'].columns)
df_yf1=df_yf['Adj Close'][ticker]
df_yf1.fillna(method='ffill',inplace=True)

for i in range(start.year,end.year+1):
    i=str(i)
    vars()["a_"+i]=df_yf1[i].dropna(thresh=10,axis=1)
    vars()["return_"+i]=np.log(vars()["a_"+i]/vars()["a_"+i].shift(1))
    vars()["return_"+i].drop(labels=vars()["return_"+i].index[0],axis=0,inplace=True)
    vars()["return_"+i].fillna(method='ffill',inplace=True)
    vars()["mean_"+i]=np.mean(vars()["return_"+i])*250
    vars()["std_"+i]=np.std(vars()["return_"+i])*(np.sqrt(250))
    vars()["skew_"+i]=vars()["return_"+i].skew()
    vars()["kurt_"+i]=vars()["return_"+i].kurt()
    vars()["corr_"+i]=vars()["return_"+i].corr()
