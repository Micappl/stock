# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:55:24 2022

@author: ant33
"""
#引入套件
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from keras.models import load_model
import xgboost
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
X_alpha = df_x_alpha.iloc[:159,:].values
#2003/8~2018/11(y)
Y_alpha = df_y_alpha.iloc[:159,:].values

#設定訓練集(Beta)
#2003/7~2018/11(不包含y)
X_beta = df_x_beta.iloc[:160,:].values
#2003/7~2018/11(y)
Y_beta = df_y_beta.iloc[:160,:].values
#設定測試集
#設定用來預測的test_data
#2018年12月的Alpha值
test_data_alpha = df_x_alpha.iloc[159:160, :].values
#2018年12月的Beta值
test_data_beta = df_x_beta.iloc[160:161, :].values



#類神經網路預測結果(Alpha)
X_alpha_ = X_alpha.astype('float64')
Y_alpha_ = Y_alpha.astype('float64')
test_data_alpha_ = test_data_alpha.astype('float64')

#丟入數據，訓練模型
model_a = Sequential()
model_a.add(Dense(32,activation='relu',input_dim=601))
model_a.add(Dense(units=32,activation='relu'))
model_a.add(Dense(units=1))
adam = optimizers.Adam(lr=0.001)
model_a.compile(optimizer = adam, loss = 'mae')
model_a.save('modela.h5')

#類神經網路預測結果(Beta)
X_beta_ = X_beta.astype('float64')
Y_beta_ = Y_beta.astype('float64')
test_data_beta_ = test_data_beta.astype('float64')

#丟入數據，訓練模型
model_b = Sequential()
model_b.add(Dense(32,activation='relu',input_dim=601))
model_b.add(Dense(units=32,activation='relu'))
model_b.add(Dense(units=1))
adam = optimizers.Adam(lr=0.001)
model_b.compile(optimizer = adam, loss = 'mae')
model_b.save('modelb.h5')

#將類神經重複預測1000次
alpha=[]
beta=[]
for _ in range(1000):
    model_a.fit(X_alpha_, Y_alpha_, batch_size=10, epochs=100)
    alpha.append(model_a.predict(test_data_alpha_))
    
    model_b.fit(X_beta_, Y_beta_, batch_size=10, epochs=100)
    beta.append(model_b.predict(test_data_beta_))

#將結果取平均(Alpha)
alpha_result = sum(alpha)/len(alpha)
#將結果取平均(Beta)ㄏ
beta_result = sum(beta)/len(beta)




