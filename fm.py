import numpy as np
from pylibfm import FM

from sklearn.feature_extraction import DictVectorizer
train = [
    {"user": "1", "item": "5", "age": 19},
    {"user": "2", "item": "43", "age": 33},
    {"user": "3", "item": "20", "age": 55},
    {"user": "4", "item": "10", "age": 20},
]
v = DictVectorizer()
X = v.fit_transform(train)
print X.toarray() 

y = np.repeat(1.0,X.shape[0])
fm = FM(learn_rate = 0.01, num_factors=10, num_iter=1,
        param_regular=(0,0,0.1))
fm.fit(X,y)
fm.predict(v.transform({"user": "1", "item": "10", "age": 24}))