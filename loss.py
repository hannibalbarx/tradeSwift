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
