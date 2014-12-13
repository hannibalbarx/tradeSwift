import pandas as pd
from sklearn.metrics import log_loss
import numpy as np
from time import strftime
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

cols=['hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'id']
coltypes=dict(zip(cols,[str]*(len(cols))))

print strftime("%a %d %b %Y %H:%M:%S")
names=['id','click','hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain','app_category','device_id','device_ip','device_model','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21']
train=pd.read_csv('data/by_day/21_22.site.csv',dtype=coltypes, names=names)
print strftime("%a %d %b %Y %H:%M:%S")

vec = DictVectorizer()

y=train['click']
del train['id']
del train['click']
del train['app_id']
del train['app_domain']
del train['app_category']
del train['device_id']
del train['device_ip']

h=train['hour']
del train['hour']
h2=h.apply(lambda a:a[6:])
train['hr']=h2

train=train.T
print strftime("%a %d %b %Y %H:%M:%S")
train=train.to_dict()
print strftime("%a %d %b %Y %H:%M:%S")
train=train.values()
X_sparse = vec.fit_transform(train)
print strftime("%a %d %b %Y %H:%M:%S")





clf= RandomForestClassifier(n_jobs=7, verbose=20)
clf = linear_model.SGDClassifier(loss='log',verbose=20, n_jobs=7)
print strftime("%a %d %b %Y %H:%M:%S")
clf.fit(X_sparse, y)
print strftime("%a %d %b %Y %H:%M:%S")

print strftime("%a %d %b %Y %H:%M:%S")
test=pd.read_csv('data/site_train.2',dtype=coltypes, names=names)
print strftime("%a %d %b %Y %H:%M:%S")

y_test=test['click']
del test['id']
del test['click']
del test['app_id']
del test['app_domain']
del test['app_category']
del test['device_id']
del test['device_ip']

h=test['hour']
del test['hour']
h2=h.apply(lambda a:a[6:])
test['hr']=h2

test=test.T
print strftime("%a %d %b %Y %H:%M:%S")
test=test.to_dict()
test=test.values()
print strftime("%a %d %b %Y %H:%M:%S")
X_test_sparse = vec.transform(test)
print strftime("%a %d %b %Y %H:%M:%S")
y_predicted=clf.predict_proba(X_test_sparse)

log_loss(np.asarray(y_test), y_predicted, eps=1e-15)