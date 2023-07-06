#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV

import pickle


# In[ ]:


def train_logistic(X, Y):
    
#start training model below...
  param_grid = [{
    'C': 0.03*np.power(3, np.arange(0,5,0.5)),
  }]

  gs = GridSearchCV(estimator = LogisticRegression(), param_grid=param_grid, scoring= 'accuracy',
                            cv=StratifiedKFold())
  gs.fit(X, Y)

# plot the separating hyperplane
  pred = gs.predict(X)
  sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y, data=X, alpha=0.5)
  sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=pred, data=X, palette=["#9b59b6", "#2ecc71"], alpha=0.5)

  model = gs.best_estimator_
  w = model.coef_[0]

  xx = np.linspace(-5, 10)
  yy = (-w[0] * xx - model.intercept_[0]) / w[1]

  plt.plot(xx,yy)

#save best_estimator to pickle
  filename = 'finalized_model.sav'
  pickle.dump(model, open(filename, 'wb'))

