#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np


# In[35]:


df= pd.read_excel('病歷資料加檢驗資料(0513).xlsx', usecols=['AC Sugar','PC Sugar','A1c','T-Cho','HDL','LDL','TG','eGFR','UACR',
'Cr'])


# In[36]:


df=df.replace(to_replace=r'\s', value=np.nan, regex=True)


# In[39]:


r, c=df.shape
#complicated generator demo
for i, j in [(i,j) for i in range(r) for j in range(c)]:
  
  if pd.isna(df.iat[i, j])!=True:
        
    try:
      pd.to_numeric(df.iat[i, j])
    except Exception as other:
      print('{'+df.iat[i, j]+'}'+'{'+str(i)+','+str(j)+'}')


# In[ ]:




