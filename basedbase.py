# coding=utf-8 
import pandas as pd
import numpy as np

from ConfigParser import SafeConfigParser
parser = SafeConfigParser()
parser.read('config.ini')
train_file = parser.get('config', 'train_file_1')
labels_file = parser.get('config', 'train_labels')
test_file = parser.get('config', 'test_file')

train_sample = pd.read_csv(train_file)
labels = pd.read_csv(labels_file)
labels.columns
train_with_labels = pd.merge(train_sample, labels, on = 'id')
train_with_labels.shape

from collections import Counter
Counter([name[0] for name in train_with_labels.columns])
del labels
del train_sample

test = pd.read_csv(test_file)

from sklearn.feature_extraction import DictVectorizer

X_numerical = []
X_test_numerical = []

vec = DictVectorizer()
names_categorical = []

train_with_labels.replace('YES', 1, inplace = True)
train_with_labels.replace('NO', 0, inplace = True)
train_with_labels.replace('nan', np.NaN, inplace = True)
test.replace('YES', 1, inplace = True)
test.replace('NO', 0, inplace = True)
test.replace('nan', np.NaN, inplace = True)

for name in train_with_labels.columns :    
    if name.startswith('x') :
        column_type, _ = max(Counter(map(lambda x: str(type(x)), train_with_labels[name])).items(), key = lambda x: x[1])
        
        # LOL expression
        if column_type == str(str) :
            train_with_labels[name] = map(str, train_with_labels[name])
            test[name] = map(str, test[name])
            names_categorical.append(name)
            print name, len(np.unique(train_with_labels[name]))
        else :
            X_numerical.append(train_with_labels[name].fillna(-999))
            X_test_numerical.append(test[name].fillna(-999))
        
X_numerical = np.column_stack(X_numerical)
X_test_numerical = np.column_stack(X_test_numerical)

X_sparse = vec.fit_transform(train_with_labels[names_categorical].T.to_dict().values())
X_test_sparse = vec.transform(test[names_categorical].T.to_dict().values())

print X_numerical.shape, X_sparse.shape, X_test_numerical.shape, X_test_sparse.shape

X_numerical = np.nan_to_num(X_numerical)
X_test_numerical = np.nan_to_num(X_test_numerical)

y_columns = [name for name in train_with_labels.columns if name.startswith('y')]

from sklearn.metrics import roc_auc_score, f1_score, log_loss, make_scorer
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier

X_numerical_base, X_numerical_meta, X_sparse_base, X_sparse_meta, y_base, y_meta = train_test_split(
    X_numerical, 
    X_sparse, 
    train_with_labels[y_columns].values,
    test_size = 0.5
)

X_meta = [] 
X_test_meta = []

print "Build meta"

for i in range(y_base.shape[1]) :
    print i
    
    y = y_base[:, i]
    if len(np.unique(y)) == 2 : 
        rf = RandomForestClassifier(n_estimators = 10, n_jobs = 1)
        rf.fit(X_numerical_base, y)
        X_meta.append(rf.predict_proba(X_numerical_meta))
        X_test_meta.append(rf.predict_proba(X_test_numerical))

        svm = LinearSVC()
        svm.fit(X_sparse_base, y)
        X_meta.append(svm.decision_function(X_sparse_meta))
        X_test_meta.append(svm.decision_function(X_test_sparse))
        
X_meta = np.column_stack(X_meta)
X_test_meta = np.column_stack(X_test_meta)

print X_meta.shape, X_numerical_meta.shape, X_test_meta.shape, X_test_numerical.shape, y_base.shape, y_meta.shape

from sklearn.externals import joblib
joblib.dump(
    (X_meta, X_numerical_meta, X_test_meta, X_test_numerical, y_base, y_meta),
    'Meta.dump',
    compress = 1,
)
