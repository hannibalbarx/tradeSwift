import pandas as pd
import numpy as np
from time import strftime

from sklearn.externals import joblib

import sys
sys.path.append('xgboost/wrapper')
import xgboost as xgb

from ConfigParser import SafeConfigParser
parser = SafeConfigParser()
parser.read('config.ini')
test_labels = pd.read_csv(parser.get('config', 'validation_labels'))
del test_labels["id"]
test_labels=np.asarray(test_labels)

meta_trees = parser.getint('config', 'meta_trees')
procs = parser.getint('config', 'procs')
print "meta trees %d"%meta_trees
print "procs %d"%procs

X_meta, X_numerical_meta, X_test_meta, X_test_numerical, y_base, y_meta = joblib.load("Meta.dump")

print X_meta.shape, X_numerical_meta.shape, X_test_meta.shape, X_test_numerical.shape, y_base.shape, y_meta.shape

X=np.hstack([X_meta, X_numerical_meta])
X_test=np.hstack([X_test_meta, X_test_numerical])

for i in range(33):
	print strftime("%a %d %b %Y %H:%M:%S")+"training label %d"%(i+1)
	dtrain=xgb.DMatrix(X, label=y_meta[:,i])
	dtest=xgb.DMatrix(X_test, label=test_labels[:,i])
	evallist  = [(dtest,'eval'), (dtrain,'train')]

	param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
	param['nthread'] = procs
	plst = param.items()
	plst += [('eval_metric', 'logloss')]
	num_round = 45
	bst = xgb.train( plst, dtrain, num_round, evallist)
