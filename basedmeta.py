# coding=utf-8 
import pandas as pd
import numpy as np

from sklearn.externals import joblib

from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

import os
from time import strftime

from ConfigParser import SafeConfigParser
parser = SafeConfigParser()
parser.read('config.ini')
test_file = parser.get('config', 'validation_file')
test = pd.read_csv(test_file)

meta_trees = parser.getint('config', 'meta_trees')
procs = parser.getint('config', 'procs')
print "meta trees %d"%meta_trees
print "procs %d"%procs

X_meta, X_numerical_meta, X_test_meta, X_test_numerical, y_base, y_meta = joblib.load("Meta.dump")

print X_meta.shape, X_numerical_meta.shape, X_test_meta.shape, X_test_numerical.shape, y_base.shape, y_meta.shape

if os.path.isfile("P.dump"):
	p_test = joblib.load("P.dump")
else:
	p_test = []

for i in range(len(p_test), y_base.shape[1]) :
    y = y_meta[:, i]
    constant = Counter(y)
    constant = constant[0] < 4 or constant[1] < 4
    predicted = None
    print strftime("%a %d %b %Y %H:%M:%S")+" working on y"+str(i+1)
    if constant :
        # Best constant
        constant_pred = np.mean(list(y_base[:, i]) + list(y_meta[:, i]))
        predicted = np.ones(X_test_meta.shape[0]) * constant_pred
    else :
        rf = RandomForestClassifier(n_estimators=meta_trees, n_jobs = procs)
        rf.fit(np.hstack([X_meta, X_numerical_meta]), y)
        predicted = rf.predict_proba(np.hstack([X_test_meta, X_test_numerical]))
        predicted = predicted[:, 1]
    p_test.append(
        predicted
    )
    joblib.dump((p_test), 'P.dump', compress = 1,)

p_test = np.column_stack(p_test)
p_test=p_test.flatten()

test_labels = pd.read_csv(parser.get('config', 'validation_labels'))
del test_labels["id"]
test_labels=np.asarray(test_labels).flatten()

print strftime("%a %d %b %Y %H:%M:%S")+" log loss = %.6f"%log_loss(test_labels, p_test, eps=1e-15)

'''
import gzip

def save_predictions(name, ids, predictions) :
    out = gzip.open(name, 'w')
    print >>out, 'id_label,pred'
    for id, id_predictions in zip(test['id'], p_test) :
        for y_id, pred in enumerate(id_predictions) :
            if pred == 0 or pred == 1 :
                pred = str(int(pred))
            else :
                pred = '%.6f' % pred
            print >>out, '%d_y%d,%s' % (id, y_id + 1, pred)

save_predictions('quick_start.csv.gz', test['id'].values, p_test)
'''
