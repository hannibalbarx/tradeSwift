'''
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
'''

from datetime import datetime
from math import log, exp, sqrt

from ConfigParser import SafeConfigParser
parser = SafeConfigParser()
parser.read('config.ini')
D = parser.getint('config', 'D')  # number of weights use for each model, we have 32 of them

lambada = parser.getfloat('config', 'lambada')  # number of weights use for each model, we have 32 of them

non_hash=list(int(z) for z in parser.get('config', 'non_hash').split(","))
hash_non_hash=parser.getboolean('config', 'hash_non_hash')
# function, generator definitions ############################################

# A. x, y generator
# INPUT:
#     path: path to train.csv or test.csv
#     label_path: (optional) path to trainLabels.csv
# YIELDS:
#     ID: id of the instance (can also acts as instance count)
#     x: a list of indices that its value is 1
#     y: (if label_path is present) label value of y1 to y33
def data(path, label_path=None, deep_hash_joins=None, hash_joins=None):
    for t, line in enumerate(open(path)):
        # initialize our generator
        if t == 0:
            # create a static x,
            # so we don't have to construct a new x for every instance
	    features_count = 146
	    if deep_hash_joins: features_count +=sum(len(x)*(len(x)-1)/2 for x in deep_hash_joins)#*2
	    if hash_joins: features_count +=len(hash_joins)#*2
	    if non_hash and hash_non_hash: features_count +=len(non_hash)
	    x = [0] * (features_count)
            if label_path:
                label = open(label_path)
                label.readline()  # we don't need the headers
            continue
        # parse x
        row = line.rstrip().split(',')
	floats=[]
        for m, feat in enumerate(row):
            if m == 0:
                ID = int(feat)
            else:
                # one-hot encode everything with hash trick
                # categorical: one-hotted
                # boolean: ONE-HOTTED
                # numerical: ONE-HOTTED!
                # note, the build in hash(), although fast is not stable,
                #       i.e., same value won't always have the same hash
                #       on different machines
		if m in non_hash: 
			floats+=[float(feat)]
			if hash_non_hash:
				x[m] = abs(hash(str(m) + '_' + feat)) % D
		else: 
			x[m] = abs(hash(str(m) + '_' + feat)) % D
        tw = 145 - 0 if hash_non_hash else len(floats)
	if deep_hash_joins:			
		for i in range(len(deep_hash_joins)):
			for j in range(len(deep_hash_joins[i])-1):
				for k in range(j+1, len(deep_hash_joins[i])):
					tw += 1
					x[tw] = abs(hash(str(tw)+"_"+row[deep_hash_joins[i][j]]+"_x_"+row[deep_hash_joins[i][k]])) % D
	if hash_joins:			
		for i in range(len(hash_joins)):
			join_str=""
			for j in range(len(hash_joins[i])-1):
				join_str+=row[hash_joins[i][j]]+"_x_"
			join_str+=row[hash_joins[i][-1]]
			tw += 1
                        x[tw] = abs(hash(str(tw)+"_"+join_str)) % D
	for i in floats:
		tw += 1
		x[tw] = i

        # parse y, if provided
        if label_path:
            # use float() to prevent future type casting, [1:] to ignore id
            y = [float(y) for y in label.readline().split(',')[1:]]
        yield (ID, x, y) if label_path else (ID, x)

# B. Bounded logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     bounded logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)

# C. Get probability estimation on x
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def predict(x, w):
    wTx = 0.
    for i in x[:len(x)-len(non_hash)]:  # do wTx
	wTx += w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
    d=0
    for i in x[len(x)-len(non_hash):]:  # do wTx
	wTx += w[D+d] * i
	d+=1
    return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid

# D. Update given model
# INPUT:
# alpha: learning rate
#     w: weights
#     n: sum of previous absolute gradients for a given feature
#        this is used for adaptive learning rate
#     x: feature, a list of indices
#     p: prediction of our model
#     y: answer
# MODIFIES:
#     w: weights
#     n: sum of past absolute gradients
def update(alpha, w, n, x, p, y):
    for i in x[:len(x)-len(non_hash)]:  # do wTx
        # do wTx
        # alpha / sqrt(n) is the adaptive learning rate
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1.
	n[i] += abs((p - y)  + lambada*w[i])
	w[i] -= ((p - y) * 1. + lambada*w[i]) * alpha / sqrt(n[i])

    d=0
    for i in x[len(x)-len(non_hash):]:  # do wTx
	n[D+d] += abs((p - y)*i  + lambada*w[D+d])
	w[D+d] -= ((p - y) * i  + lambada*w[D+d]) * alpha / sqrt(n[D+d])
	d+=1

def update_floats(alpha, w, n, x, p, y):
    global D,non_hash
    d=0
    for i in x[len(x)-len(non_hash):]:  # do wTx
	n[D+d] += abs((p - y)*i  + lambada*w[D+d])
	w[D+d] -= ((p - y) * i   + lambada*w[D+d]) * alpha / sqrt(n[D+d])
	d+=1
