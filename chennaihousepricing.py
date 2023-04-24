# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv('Chennai houseing sale.csv')

dataset.head()

dataset.shape

dataset.columns

dataset.info()

dataset.describe()

dataset.select_dtypes(include=['int64','float64']).columns

"""Dealing with null values"""

dataset.isnull().values.any()

#if null remove null rows

dataset.isnull().values.sum()

dataset.isnull().sum()

#null values with heatmap
plt.figure(figsize=(16,9))
sns.heatmap(dataset.isnull())
plt.show()

"""Qs_Overall cleaning data"""

dataset.columns[dataset.isnull().any()]

dataset['QS_OVERALL']=dataset['QS_OVERALL'].fillna(dataset['QS_OVERALL'].mean())
dataset['N_BEDROOM']=dataset['N_BEDROOM'].fillna(dataset['N_BEDROOM'].mean())
dataset['N_BATHROOM']=dataset['N_BATHROOM'].fillna(dataset['N_BATHROOM'].mean())

len(dataset.columns[dataset.isnull().any()])

dataset3=dataset.select_dtypes(include=['int64','float64'])
dataset.isnull().values.any()
xf=dataset3.drop('SALES_PRICE',axis=1)
yf=dataset3['SALES_PRICE']
xf.head()
yf.head()
##Distplot

plt.figure(figsize=(16,9))
bar=sns.histplot(dataset['SALES_PRICE'])
bar.legend(["skewness:{:.2f}".format(dataset['SALES_PRICE'].skew())])
plt.show()

#correlational matrix

dataset_2=dataset.drop(columns='SALES_PRICE')
dataset_2.corrwith(dataset['SALES_PRICE']).plot.bar(figsize=(16,9),title='correlation with SALES_PRICE',grid=True)

#As we can see my the correlational matrix that the Reg_fee has highest co relation with the SALES_PRICE

## dealing with categorical values

dataset.select_dtypes(include='object').columns

dataset=pd.get_dummies(data=dataset,drop_first=True)

x=dataset.drop(columns='SALES_PRICE')
y=dataset['SALES_PRICE']
print(x.head)
print(y.head)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

x_train.shape

y_train.shape

x_test.shape

y_test.shape


from sklearn.preprocessing import StandardScaler 
ss=StandardScaler()

x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)

x_train

x_test

from sklearn.linear_model import LinearRegression

regressor_mlr=LinearRegression()
regressor_mlr.fit(x_train,y_train)

y_pred=regressor_mlr.predict(x_test)

from sklearn.metrics import r2_score

r2_score(y_test,y_pred)

from sklearn.ensemble import RandomForestRegressor

r_ref=RandomForestRegressor()
r_ref.fit(x_train,y_train)

y_pred=r_ref.predict(x_test)

r2_score(y_test,y_pred)
r_ref=RandomForestRegressor()
'''so we know that random regresssor forest is the best one'''
r_ref.fit(xf,yf)


import pickle

pickle.dump(r_ref,open('model.pkl','wb'))

r_ref=pickle.load(open('model.pkl','rb'))
print(r_ref.predict([[1984,26,1,1,3,4,4,4,5,760000,144400]]))

