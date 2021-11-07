#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[36]:


df = pd.read_csv('kc_house_data.csv', usecols=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'price', 'yr_built'])


# In[37]:


df.head()


# In[10]:


## Exloration Data Analysis


# In[38]:


df.shape


# In[39]:


df.info()


# In[40]:


df.describe()


# In[41]:


sns.displot(df['price'])


# In[16]:


df['bathrooms'] = df['bathrooms'].astype('int')
df['bedrooms'] = df['bedrooms'].replace(33,3)


# In[42]:


df.isnull().sum()


# In[43]:


numeric_features = df.dtypes[df.dtypes != 'Object'].index


# In[44]:


numeric_features


# In[45]:


cormat = df[numeric_features].corr()
plt.subplots(figsize= (12,10))
sns.heatmap(cormat, vmax = 0.9, square = True)


# In[46]:


df.corr().style.background_gradient().set_precision(2)


# In[47]:


x = df.drop(columns='price')
y = df['price'] 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)           


# In[48]:


linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)
print(linear_reg.coef_)
print(linear_reg.intercept_)


# In[49]:


linear_reg.score(x_test, y_test)


# In[50]:


linear_reg.predict([[4,2,2500,8,2001]])


# In[ ]:




