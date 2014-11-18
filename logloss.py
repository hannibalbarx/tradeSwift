import pandas as pd
import numpy as np
from sklearn.externals import joblib
from ConfigParser import SafeConfigParser
parser = SafeConfigParser()
parser.read('config.ini')

test_labels = pd.read_csv(parser.get('config', 'validation_labels'))
del test_labels["id"]
test_labels=np.asarray(test_labels).flatten()

p_test = joblib.load("P.dump")
p_test = np.column_stack(p_test)
p_test=p_test.flatten()

for i in range(len(p_test)):
 if p_test[i]<1e-15: p_test[i]=1e-15
 elif p_test[i]>1-1e-15: p_test[i]=1-1e-15

p_test_log=np.log(p_test)
d=test_labels.T.dot(p_test_log)
p_test_log_1=np.log(1-p_test)
d_1=(1-test_labels).T.dot(p_test_log_1)
print "log loss = %.6f"%((-d-d_1)/len(p_test))
