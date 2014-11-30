# coding=utf-8 
import pandas as pd
import numpy as np

data_dir = 'data/'
train = pd.read_csv(data_dir + 't.rfin.site_train')

y=train["click"]
del train["click"]

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=20, verbose=20)

gbc.fit(train,y)

