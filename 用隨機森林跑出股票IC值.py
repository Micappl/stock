# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:28:16 2022

@author: ant33
"""
#引入套件
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from keras.models import load_model


#指定路徑
path = r'C:\Users\ant33\OneDrive\桌面'

#改變當前工作目錄
os.chdir(path)

#刪完資料的Alpha工作表
df_x_alpha = pd.read_excel("1Q_Chg._Alpha_Monthly.xlsm", sheet_name='Alpha', header = 3,index_col = 2)
df_x_alpha = df_x_alpha.iloc[:,2:-1].dropna()

#Alpha IC值
df_y_alpha = pd.read_excel("1Q_Chg._Alpha_Monthly.xlsm", sheet_name='Alpha IC', header = 3,index_col = 2)
df_y_alpha = df_y_alpha.dropna(axis=1)


#刪完資料的Beta工作表
df_x_beta = pd.read_excel("1Q Beta_Monthly.xlsm", sheet_name='Beta', header = 3,index_col = 2)
df_x_beta = df_x_beta.dropna(axis=1).iloc[:,2:]

#Beta IC值
df_y_beta = pd.read_excel("1Q Beta_Monthly.xlsm", sheet_name='Beta IC', header = 3,index_col = 2)
df_y_beta = df_y_beta.dropna(axis=1)

#轉置(XY軸對調)
df_x_alpha = df_x_alpha.T
df_y_alpha = df_y_alpha.T

df_x_beta = df_x_beta.T
df_y_beta = df_y_beta.T

#設定訓練集(Alpha)
#2003/8~2018/11(不包含y)
X_alpha = df_x_alpha.iloc[:183,:].values
#2003/8~2018/11(y)
Y_alpha = df_y_alpha.iloc[:183,:].values


#設定訓練集(Beta)
#2003/7~2018/11(不包含y)
X_beta = df_x_beta.iloc[:184,:].values
#2003/7~2018/11(y)
Y_beta = df_y_beta.iloc[:184,:].values


#設定測試集
#設定用來預測的test_data
#2018年12月的Alpha值
test_data_alpha = df_x_alpha.iloc[183:184, :].values
test_data_beta = df_x_beta.iloc[184:185, :].values

#建立模型
regr = RandomForestRegressor(max_depth=2, n_estimators=100)
xgb = XGBClassifier()
reg = LinearRegression()
svr = SVR(kernel='linear')
gbr = GradientBoostingRegressor()

#丟入數據(Alpha)
regr.fit(X_alpha,Y_alpha)
xgb.fit(X_alpha,Y_alpha)
reg.fit(X_alpha,Y_alpha)
svr.fit(X_alpha,Y_alpha)
gbr.fit(X_alpha,Y_alpha)

#預測(Alpha)
print('alpha')
print('======================')
print('隨機森林', regr.predict(test_data_alpha))
print('xgboost', xgb.predict(test_data_alpha))
print('線性回歸',reg.predict(test_data_alpha))
print('SVR',svr.predict(test_data_alpha))
print('GBR',gbr.predict(test_data_alpha))

#丟入數據(Beta)
regr.fit(X_beta,Y_beta)
xgb.fit(X_beta,Y_beta)
reg.fit(X_beta,Y_beta)
svr.fit(X_beta,Y_beta)
gbr.fit(X_beta,Y_beta)

#預測(Beta)
print('beta')
print('======================')
print('隨機森林', regr.predict(test_data_beta))
print('xgboost', xgb.predict(test_data_beta))
print('線性回歸',reg.predict(test_data_beta))
print('SVR',svr.predict(test_data_beta))
print('GBR',gbr.predict(test_data_beta))


#將隨機森林重複預測1000次
alpharandom=[]
betarandom=[]
for _ in range(1000):
    regr.fit(X_alpha,Y_alpha)
    alpharandom.append(regr.predict(test_data_alpha))
    
    regr.fit(X_beta,Y_beta)
    betarandom.append(regr.predict(test_data_beta))

alpharandom_result = sum(alpharandom)/len(alpharandom)

betarandom_result = sum(betarandom)/len(betarandom)

