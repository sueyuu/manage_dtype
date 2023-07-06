#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from collections import Counter
from collections import defaultdict

from logistic_model import train_logistic
import pickle


# In[2]:


def voting_dtypes(_x, dtype_counter, str_set, float_set):
#first remove ><
  x = _x.replace(to_replace=r'^[><](?=([0-9]+\.?[0-9]*$)|([0-9]*\.[0-9]+$))', value='', regex=True)
  for col in x.columns:
    total_dtypes=[]
    str_form=[]
    float_form=[]
    for rw in range(x[col].index[0], x[col].index[-1]+1):
      try:
#first get nan(do not pass into dtypesdict)
        if pd.isnull(x.at[rw, col]):
          pass
#then if not float
        else:
          x.at[rw, col] = float(x.at[rw, col])
          total_dtypes.append('float')
          float_form.append(x.at[rw, col])
      except:
#then should be string
        x.at[rw, col] = str(x.at[rw, col])
        total_dtypes.append('str')
        str_form.append(x.at[rw, col])
    dtype_counter[col].update({ k: v + dtype_counter[col][k] for k, v in Counter(total_dtypes).items()})
     
    str_set[col]=str_set[col]|set(str_form)
    float_set[col]=float_set[col]|set(float_form)


# In[3]:


def manage_dtypes(_path, start, end = None):
  df = pd.read_csv(_path, chunksize=10000, dtype=str)

  df_str=defaultdict(set)
  df_float=defaultdict(set)
  df_dict=defaultdict(lambda: defaultdict(int))

#too many columns, could only save chunk into chunks and then concat and save
  chunks = []

  for chunk in df:
    
#空格replace with ''
    chunk.loc[:, start:end] = chunk.loc[:, start:end].replace(to_replace=r'\s', value='', regex=True)
#replace空字串with nan
    chunk.loc[:, start:end] = chunk.loc[:, start:end].replace(to_replace=r'^\s*$', value=np.nan, regex=True)
  
    voting_dtypes(chunk.loc[:, start:end], df_dict, df_str, df_float)
    
    chunks.append(chunk)

#rewrite managed chunk to origin csv    
  pd.concat(chunks, ignore_index=True).to_csv(_path, index=False)
    
#return dtypes_counter, string_set, float_set
  return df_dict, df_str, df_float


# In[68]:


labeling=[1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,
          1,0,0,1,1,0,0,0,0,1,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,0,0,0,0,1,0,1,
         1,0,0,0,0,1,1,1,0,1,0,0,0,1,1]

df_dict, df_str, df_float = manage_dtypes('病歷資料加檢驗資料.csv', 'AC Sugar')

N=[[],[]]
Y=[]
    
for i, (col, _type) in enumerate(df_dict.items()):
  N[0].append(_type['float']+0.001)
  N[1].append(_type['str']+0.001)
  Y.append(labeling[i])

X=np.transpose(np.log(N))

Y=np.array(Y)

train_logistic(X, Y)


# In[4]:


df_dict, df_str, df_float = manage_dtypes('2005_202106病歷資料加檢驗資料(0830).csv', 'AC Sugar')
N=[[],[]]
    
for i, (col, _type) in enumerate(df_dict.items()):
  N[0].append(_type['float']+0.001)
  N[1].append(_type['str']+0.001)

X=np.transpose(np.log(N))


# In[5]:


# load the model from disk
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

result= loaded_model.predict(X)

result = [bool(i) for i in result]


# In[6]:


def check_changed(origin, changed):
  originlist =  []
#check 'inf' and return origin value
  for col in changed.loc[:, result]:
    
    for rw in range(origin.index[0], origin.index[-1]+1):

      if changed.at[rw, col] == float('inf'):

        originlist.append(origin.at[rw, col])

        continue
      try:
        if re.match(r'^[><]', changed.at[rw, col]):
          print(f'><:{changed.at[rw, col]}')
      except:
        pass
  return originlist 


# In[7]:


#manage datas according to predicted dtypes
df= pd.read_csv('2005_202106病歷資料加檢驗資料(0830).csv', chunksize = 10000, dtype=str)

chunks = []

for chunk in df:  
    
#if classify as float type
#非><*數字*.數字者*replace with inf

#to avoid SettingWithCopyWarning, assign new df with copy
  _df = chunk.loc[:, 'AC Sugar':].copy()

  _df.loc[:, result] = _df.loc[:, result].replace(
      to_replace = r'^(?![><]?(([0-9]+\.?[0-9]*$)|([0-9]*\.[0-9]+$)))', value=float('inf'), regex=True)

  chunk.update(_df)

  chunks.append(chunk)
#concat and transfer dtypes    
pd.concat(chunks, ignore_index=True).astype(
    {list(df_dict.keys())[i]: 'float64' if boo else 'str' for i, boo in enumerate(result)}, errors = 'ignore').to_csv(
    '2005_202106病歷資料加檢驗資料(0830).csv', index=False, chunksize=10000, encoding = 'utf-8-sig')


# In[ ]:




