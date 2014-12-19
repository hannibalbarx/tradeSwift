import pandas as pd
import numpy as np
import sys
from time import strftime
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib

cols=['id', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
coltypes=dict(zip(cols,[str]*(len(cols))))

print strftime("%a %d %b %Y %H:%M:%S ")+"loading dictvect"
vec=joblib.load('test_app_dictvectorizer.dump')

print strftime("%a %d %b %Y %H:%M:%S ")+"loading train file"
train=pd.read_csv('data/site_train',dtype=coltypes)
print strftime("%a %d %b %Y %H:%M:%S ")+"done"

y=train['click']
del train['id']
del train['click']
del train['site_id']
del train['site_domain']
del train['site_category']

h=train['hour']
del train['hour']
h2=h.apply(lambda a:a[6:])
train['hour']=h2

print strftime("%a %d %b %Y %H:%M:%S ")+"transform"
train=train.T
print strftime("%a %d %b %Y %H:%M:%S ")+"dict"
train=train.to_dict()
print strftime("%a %d %b %Y %H:%M:%S ")+"dictvect transform"
train=train.values()
X_sparse = vec.transform(train)
print strftime("%a %d %b %Y %H:%M:%S ")+"done"

print strftime("%a %d %b %Y %H:%M:%S ")+"saving file"
f=open("data/fm.app_train","wb")
for j in range(len(y)):
	a=X_sparse.getrow(j).nonzero()
	s=str(y[j])
	for i in a[1]:
		s+=" %d:1"%i
	f.write(s+"\n")

print strftime("%a %d %b %Y %H:%M:%S ")+"done"
