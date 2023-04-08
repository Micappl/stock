# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:00:10 2022

@author: ant33
"""
#1.算每年的年化報酬率、報酬率標準差、報酬率偏態係數、報酬率峰態係數、相關係數 
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


#1.算α、β
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
ticker1=[str(i)+'.TW' for i in df['證券名稱']]
ticker1.append('^TWII')
df_yf2=yf.download(ticker1,start=start,end=end)
n_close1=len(df_yf2['Adj Close'].columns)
df_yf2=df_yf2['Adj Close'][ticker1]
df_yf2.fillna(method='ffill',inplace=True)

def CAPM(df_return):
    capm_out = {}
    for company in df_return.columns[:n_close-1]:
        if df_return[company].isna().sum()<len(df_return.index):
           df_return[company]=df_return[company]-rf 
           df_return['^TWII']=df_return['^TWII']-rf
           data_df=pd.concat([df_return[company],df_return['^TWII']],axis=1)
           data_df.columns=[company,'index']
           X=data_df[['index']].assign(Intercept=1)
           Y=data_df[company]
           X=sm.add_constant(X)
           model=sm.OLS(Y,X)
           results=model.fit()
           Alpha=results.params[1]
           Beta=results.params[0]
           capm_out[company]=[Alpha, Beta]
        else:
           Alpha=np.nan
           Beta=np.nan
           capm_out[company]=[Alpha,Beta]
          
    df_ab=pd.DataFrame.from_dict(capm_out, orient='index',columns=['Alpha','Beta'])
    return df_ab

for i in range(start.year,end.year+1):
    i=str(i)
    vars()["a_"+i]=df_yf2[i].dropna(thresh=50,axis=1)
    vars()["return_"+i]=np.log(vars()["a_"+i]/vars()["a_"+i].shift(1))
    vars()["return_"+i].drop(labels=vars()["return_"+i].index[0],axis=0,inplace=True)
    vars()["return_"+i].fillna(method='ffill',inplace=True)
    vars()["CAPM_"+i]=CAPM(vars()["return_"+i])



writer = pd.ExcelWriter("HW_A02_Robo_09155243.xlsx",engine = 'xlsxwriter')

for i in range(start.year,end.year+1):
    i = str(i)
    vars()["CAPM_"+i].to_excel(writer,sheet_name = "CAPM_"+i)
writer.save()



