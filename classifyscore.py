# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 11:08:30 2018

@author: Denizhan Akar
"""

#import pickle
#from sklearn.naive_bayes import MultinomialNB, BernoulliNB
#from sklearn.linear_model import LogisticRegression, SGDClassifier
#from sklearn.svm import SVC, LinearSVC, NuSVC
#from statistics import mode

import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np


data = pd.read_csv("C:/Users/HP/Desktop/Likelihood_to_puchase_sample_data.csv", engine="python")

figure = plt.figure(figsize=(30,8))
plt.hist([data[data['current_is_sale']==1]['last_1_day_session_count'], data[data['current_is_sale']==0]['last_1_day_session_count']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('last_1_day_session_count')
plt.ylabel('Numbers sold')
plt.legend()

embarked_dummies = pd.get_dummies(data['referrer'],prefix='referrer')
data = pd.concat([data,embarked_dummies],axis=1)
data.drop('referrer',axis=1,inplace=True)
data.drop('date',axis=1,inplace=True)


# Modelling
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score



def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)



def recover_train_test_target():
    global data
    
    
    targets = data.current_is_sale
    train = data.head(800000)
    test = data.iloc[800000:]
    
    return train, test, targets


train, test, targets = recover_train_test_target()


# Compute the importance of features, destroy irrelevant
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)


features.plot(kind='barh', figsize=(20, 20))

















