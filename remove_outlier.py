#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


def boxplot_fill(col):

  iqr = col.quantile(0.75)-col.quantile(0.25)

  u_th = col.quantile(0.75) + 1.5*iqr
  l_th = col.quantile(0.25) - 1.5*iqr
  
  return col.apply(lambda x: x if x > l_th and x < u_th else np.nan)

