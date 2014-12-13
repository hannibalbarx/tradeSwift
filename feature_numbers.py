import pandas as pd
import numpy as np
from time import strftime
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()

cols=['hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
coltypes=dict(zip(cols,[str]*(len(cols))))

names_test=['hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain','app_category','device_id','device_ip','device_model','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21']
test=pd.read_csv('fn.site_test',dtype=coltypes, names=names_test)
print strftime("%a %d %b %Y %H:%M:%S")

#del test['site_id']
#del test['site_domain']
#del test['site_category']
del test['app_id']
del test['app_domain']
del test['app_category']

test=test.T
print strftime("%a %d %b %Y %H:%M:%S")
test=test.to_dict()
test=test.values()
print strftime("%a %d %b %Y %H:%M:%S")
X_test_sparse = vec.fit_transform(test)
print strftime("%a %d %b %Y %H:%M:%S")

f=open("dv.fn.site_test","wb")
for j in range(X_test_sparse.shape[0]):
	a=X_test_sparse.getrow(j).nonzero()
	s=""
	for i in a[1]:
		s+="%d,"%i
	f.write(s[:len(s)-1]+"\n")

print strftime("%a %d %b %Y %H:%M:%S")

names=['click','hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain','app_category','device_id','device_ip','device_model','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21']
train=pd.read_csv('fn.site_train',dtype=coltypes, names=names)
print strftime("%a %d %b %Y %H:%M:%S")

y=train['click']
del train['click']
#del train['site_id']
#del train['site_domain']
#del train['site_category']
del train['app_id']
del train['app_domain']
del train['app_category']

train=train.T
print strftime("%a %d %b %Y %H:%M:%S")
train=train.to_dict()
print strftime("%a %d %b %Y %H:%M:%S")
train=train.values()
X_sparse = vec.transform(train)
print strftime("%a %d %b %Y %H:%M:%S")

print strftime("%a %d %b %Y %H:%M:%S")
f=open("dv.fn.site_train","wb")
for j in range(len(y)):
	a=X_sparse.getrow(j).nonzero()
	s=str(y[j])
	for i in a[1]:
		s+=",%d"%i
	f.write(s+"\n")

print strftime("%a %d %b %Y %H:%M:%S")

