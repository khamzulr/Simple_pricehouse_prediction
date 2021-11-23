#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('kc_house_data.csv', usecols=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'price', 'yr_built'])
df.head()


## Exloration Data Analysis
df.shape

df.info()

df.describe()

sns.displot(df['price'])



df['bathrooms'] = df['bathrooms'].astype('int')
df['bedrooms'] = df['bedrooms'].replace(33,3)

df.isnull().sum()

numeric_features = df.dtypes[df.dtypes != 'Object'].index
numeric_features

cormat = df[numeric_features].corr()
plt.subplots(figsize= (12,10))
sns.heatmap(cormat, vmax = 0.9, square = True)

df.corr().style.background_gradient().set_precision(2)

x = df.drop(columns='price')
y = df['price'] 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)           


linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)
print(linear_reg.coef_)
print(linear_reg.intercept_)


linear_reg.score(x_test, y_test)
linear_reg.predict([[4,2,2500,8,2001]])
