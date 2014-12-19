import pandas as pd
import numpy as np
import sys
from time import strftime
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib

cols=['id', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
coltypes=dict(zip(cols,[str]*(len(cols))))

print strftime("%a %d %b %Y %H:%M:%S")
test=pd.read_csv('data/app_test',dtype=coltypes)
print strftime("%a %d %b %Y %H:%M:%S")

h=test['hour']
del test['hour']
h2=h.apply(lambda a:a[6:])
test['hour']=h2

del test['id']
del test['site_id']
del test['site_domain']
del test['site_category']

vec = DictVectorizer()

print strftime("%a %d %b %Y %H:%M:%S")
test=test.T
print strftime("%a %d %b %Y %H:%M:%S")
test=test.to_dict()
test=test.values()
print strftime("%a %d %b %Y %H:%M:%S")
X_test_sparse = vec.fit_transform(test)
print strftime("%a %d %b %Y %H:%M:%S")

joblib.dump(vec, 'test_app_dictvectorizer.dump', compress = 1,)

f=open("data/fm.app_test","wb")
for j in range(X_test_sparse.shape[0]):
	a=X_test_sparse.getrow(j).nonzero()
	s="0"
	for i in a[1]:
		s+=" %d:1"%i
	f.write(s+"\n")

print strftime("%a %d %b %Y %H:%M:%S")














print strftime("%a %d %b %Y %H:%M:%S")
names=['id','click','hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain','app_category','device_id','device_ip','device_model','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21']
train=pd.read_csv('data/site_train',dtype=coltypes, names=names)
print strftime("%a %d %b %Y %H:%M:%S")

vec = DictVectorizer()

y=train['click']
del train['id']
del train['click']
del train['site_id']
del train['site_domain']
del train['site_category']
#del train['device_id']
#del train['device_ip']

h=train['hour']
del train['hour']
h2=h.apply(lambda a:a[6:])
train['hour']=h2

print strftime("%a %d %b %Y %H:%M:%S")
train=train.T
print strftime("%a %d %b %Y %H:%M:%S")
train=train.to_dict()
print strftime("%a %d %b %Y %H:%M:%S")
train=train.values()
X_sparse = vec.transform(train)
print strftime("%a %d %b %Y %H:%M:%S")

print strftime("%a %d %b %Y %H:%M:%S")
f=open("data/fm.app_train","wb")
for j in range(len(y)):
	a=X_sparse.getrow(j).nonzero()
	s=str(y[j])
	for i in a[1]:
		s+=" %d:1"%i
	f.write(s+"\n")

print strftime("%a %d %b %Y %H:%M:%S")










names_test=['id','hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain','app_category','device_id','device_ip','device_model','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21']
test=pd.read_csv('data/site_test',dtype=coltypes, names=names_test)
print strftime("%a %d %b %Y %H:%M:%S")

#y_test=test['click']
del test['id']
#del test['click']
del test['app_id']
del test['app_domain']
del test['app_category']
#del test['device_id']
#del test['device_ip']

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

f=open("data/fm.site_test","wb")
for j in range(X_test_sparse.shape[0]):
	a=X_test_sparse.getrow(j).nonzero()
	s="0"
	for i in a[1]:
		s+=" %d:1"%i
	f.write(s+"\n")

print strftime("%a %d %b %Y %H:%M:%S")





import pandas as pd
from sklearn.metrics import log_loss
import numpy as np
import sys
from time import strftime
from sklearn.feature_extraction import DictVectorizer

cols=['id', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
coltypes=dict(zip(cols,[str]*(len(cols))))

print strftime("%a %d %b %Y %H:%M:%S")
names=['id','click','hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain','app_category','device_id','device_ip','device_model','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21']

test=pd.read_csv('data/by_day/29_30.site.csv',dtype=coltypes, names=names)
y_test=np.asarray(test['click'])

yp_i_05=np.asarray(pd.read_csv("logs/MCMC_i.05_sitefeminabuddha.d8.csv", names=["p"])['p'])
log_loss(y_test, yp_i_05, eps=1e-15)

y_f12i5=np.asarray(pd.read_csv("logs/MCMC_f12i5_sitefeminabuddha.d8.csv", names=["p"])['p'])
log_loss(y_test,y_f12i5, eps=1e-15)




import pandas as pd
p1=pd.read_csv("siteminabuddha.14Dec1625.0.csv")
p2=pd.read_csv("sitefeminabuddha.17Dec1250.csv")
ids=p1['id']
p1=p1['click']
p2=p2['click']
p=(p1+p2)/2
f=open("siteminabuddha.17Dec2121.csv","wb")
f.write("id,click\n")
for i in range(2858159+1):
 f.write("%s,%.8f\n"%(ids[i],p[i]))
f.close()
