import pandas as pd
import numpy as np
from sklearn.externals import joblib
from ConfigParser import SafeConfigParser
parser = SafeConfigParser()
parser.read('config.ini')

test_labels = pd.read_csv(parser.get('config', 'validation_labels'))
del test_labels["id"]

p_test = joblib.load("P.dump")
p_test = np.column_stack(p_test)
p_test=p_test.flatten()

for j in range(len(p_test)):
	for i in range(len(p_test[j])):
	 if p_test[j][i]<1e-15: p_test[j][i]=1e-15
	 elif p_test[j][i]>1-1e-15: p_test[j][i]=1-1e-15

for i in range(17):
	test_labels2=np.asarray(test_labels["y"+str(i+1)])
	p_test_log=np.log(p_test[:,i])
	d=test_labels2.T.dot(p_test_log)
	p_test_log_1=np.log(1-p_test[:,i])
	d_1=(1-test_labels2).T.dot(p_test_log_1)
	print "log loss %d = %.6f"%(i,((-d-d_1)/len(p_test)))
