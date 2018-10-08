# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:57:52 2018

@author: HP
"""

import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array([0, 0, 1, 0,0,0,0,0])
#y_scores = np.array([0.1, 0.4, 0.35, 0.8])
y_scores = np.array([0,0,1,0,0,0,0,0])
roc_auc_score(y_true, y_scores)
