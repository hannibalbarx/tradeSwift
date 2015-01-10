#logloss
import pandas as pd
from sklearn.metrics import log_loss
import numpy as np
from time import strftime
y=pd.read_csv('../cv_results/capp_train')
y=y['click']
y_pred=pd.read_csv('29_30.out',names=['click'])
y_pred=y_pred['click']
print log_loss(np.asarray(y)[11242146:14596137],np.asarray(y_pred),eps=1e-15)
y_pred=pd.read_csv('23_24.out',names=['click'])
y_pred=y_pred['click']
print log_loss(np.asarray(y)[3642468:5965405],np.asarray(y_pred),eps=1e-15)
y_pred=pd.read_csv('25_26.out',names=['click'])
y_pred=y_pred['click']
print log_loss(np.asarray(y)[5965405:8201223],np.asarray(y_pred),eps=1e-15)
y_pred=pd.read_csv('27_28.out',names=['click'])
y_pred=y_pred['click']
print log_loss(np.asarray(y)[8201223:11242146],np.asarray(y_pred),eps=1e-15)
y_pred=pd.read_csv('29_30.out',names=['click'])
y_pred=y_pred['click']
print log_loss(np.asarray(y)[11242146:14596137],np.asarray(y_pred),eps=1e-15)


import pandas as pd
from sklearn.metrics import log_loss
import numpy as np
from time import strftime
y=np.asarray(pd.read_csv('apptrainbuddha.csv')['click'])
y_f_=np.asarray(pd.read_csv('apptrainbuddha.f_.26Dec2034.csv')['click'])
y_fm=np.asarray(pd.read_csv('apptrainbuddha.fm.26Dec1300.csv')['click'])
log_loss(y, y_f_, eps=1e-15)
log_loss(y, y_fm, eps=1e-15)

y=np.asarray(pd.read_csv('sitetrainbuddha.csv')['click'])
y_f_=np.asarray(pd.read_csv('sitetrainbuddha.f_.26Dec2015.csv')['click'])
y_fm=np.asarray(pd.read_csv('sitetrainbuddha.fm.26Dec1330.csv')['click'])
f=open("siteminabuddha.17Dec2121.csv","wb")
f.write("id,click\n")
for i in range(2858159+1):
	f.write("%s,%.8f\n"%(ids[i],p[i]))


import pandas as pd
import numpy as np

y=np.asarray(pd.read_csv('apptrainbuddha.csv')['click'])
y_f_=np.asarray(pd.read_csv('apptrainbuddha.f_.26Dec2034.csv')['click'])
y_fm=np.asarray(pd.read_csv('apptrainbuddha.fm.26Dec1300.csv')['click'])
y_xg=np.asarray(pd.read_csv('apptrainbuddha.xg.29Dec1630.csv')['click'])

b=0
e=len(y)/2+1
def loss(theta):
	return -sum(y[b:e]*np.log((theta[0]*y_f_[b:e]+theta[1]*y_fm[b:e]+theta[2]*y_xg[b:e])/sum(theta))+(1-y[b:e])*np.log(1-((theta[0]*y_f_[b:e]+theta[1]*y_fm[b:e]+theta[2]*y_xg[b:e])/sum(theta))))/len(y[b:e])

theta_min = minimize(loss, theta, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})


import pandas as pd
import numpy as np
y_f_=np.asarray(pd.read_csv('appminabuddha.15Dec1702.0.csv')['click'])
y_xg=np.asarray(pd.read_csv('appminabuddha.28Dec1830.csv')['click'])
y=(y_f_*.64+y_xg*0.75)/(.64+.75)
f=open("appminabuddha.30Dec1400.csv","wb")
f.write("id,click\n")
ids=np.asarray(pd.read_csv('appminabuddha.15Dec1702.0.csv')['id'])
for i in range(len(ids)):
	f.write("%s,%.8f\n"%(ids[i],y[i]))
f.close()

for i in range(len(y_max)):
 h=max(y_f_[i],y_xg[i])
 l=min(y_f_[i],y_xg[i])
 if (1.0-h>l):
  y_max[i]=h
 else:
  y_max[i]=l
