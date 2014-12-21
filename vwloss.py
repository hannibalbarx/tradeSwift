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
