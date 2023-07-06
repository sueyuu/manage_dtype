#!/usr/bin/env python
# coding: utf-8

# In[177]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from remove_outlier import boxplot_fill

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer


# In[178]:


sns.set()


# In[200]:


df_copy=pd.read_csv('all.csv')


# In[206]:


df = df_copy.copy()


# In[207]:


df.skew(axis=0)


# In[208]:


df.kurtosis(axis=0)


# In[210]:


fig, axes = plt.subplots(1, 2, figsize=(8, 6))
sns.histplot(ax=axes[0], data=[df.sbp, df.dbp], kde=True)
sns.boxplot(ax=axes[1], data=[df.sbp, df.dbp])


# In[211]:


s_=df.sbp.to_numpy().reshape(-1, 1)
d_=df.dbp.to_numpy().reshape(-1, 1)


# In[212]:


bc = PowerTransformer(method='box-cox')
yj = PowerTransformer(method='yeo-johnson')
qt1 = QuantileTransformer(output_distribution='normal', random_state=0)
qt2 = QuantileTransformer(output_distribution='normal', random_state=0)
s_qt=qt1.fit_transform(s_)
d_qt=qt2.fit_transform(d_)
s_bc=bc.fit_transform(s_)
d_bc=bc.fit_transform(d_)
s_yj=yj.fit_transform(s_)
d_yj=yj.fit_transform(d_)


# In[213]:


fig, axes = plt.subplots(3, 2, figsize=(12, 12))
sns.histplot(ax=axes[0, 0], data=s_qt, kde=True)
sns.histplot(ax=axes[0, 1], data=d_qt, kde=True)
sns.histplot(ax=axes[1, 0], data=s_bc, kde=True)
sns.histplot(ax=axes[1, 1], data=d_bc, kde=True)
sns.histplot(ax=axes[2, 0], data=s_yj, kde=True)
sns.histplot(ax=axes[2, 1], data=d_yj, kde=True)


# In[214]:


fig, axes = plt.subplots(3, 2, figsize=(8, 12))
sns.boxplot(ax=axes[0, 0], data=s_qt)
sns.boxplot(ax=axes[0, 1], data=d_qt)
sns.boxplot(ax=axes[1, 0], data=s_bc)
sns.boxplot(ax=axes[1, 1], data=d_bc)
sns.boxplot(ax=axes[2, 0], data=s_yj)
sns.boxplot(ax=axes[2, 1], data=d_yj)


# In[215]:


s_ar=qt1.inverse_transform(boxplot_fill(pd.Series(s_qt.reshape(1,-1)[0])).to_numpy().reshape(-1, 1))
d_ar=qt2.inverse_transform(boxplot_fill(pd.Series(d_qt.reshape(1,-1)[0])).to_numpy().reshape(-1, 1))


# In[216]:


fig, axes = plt.subplots(1, 2, figsize=(8, 6))

sns.histplot(ax=axes[0], data=[df_copy.sbp-10, s_ar.reshape(1,-1)[0]], kde=True)
sns.histplot(ax=axes[1], data=[df_copy.dbp-10, d_ar.reshape(1,-1)[0]], kde=True)


# In[217]:


df.sbp=pd.Series(s_ar.reshape(1,-1)[0])
df.dbp=pd.Series(d_ar.reshape(1,-1)[0])


# In[218]:


df_copy.describe()


# In[219]:


df.describe()


# In[220]:


df.skew(axis=0)


# In[221]:


df.kurtosis(axis=0)


# In[196]:


df.to_csv('all.csv', index=False)


# In[ ]:




