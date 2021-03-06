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
from time import strftime
from math import log, exp, sqrt

from ConfigParser import SafeConfigParser
parser = SafeConfigParser()
parser.read('config.ini')
D = parser.getint('config', 'D')  # number of weights use for each model, we have 32 of them

lambada = parser.getfloat('config', 'lambada')  # number of weights use for each model, we have 32 of them

deep_hash_joins=[]
hash_joins=[]
if parser.has_option('config', 'deep_hash_joins'):
        deep_hash_joins=list(list(int(z) for z in y.split(",")) for y in list(x for x in parser.get('config', 'deep_hash_joins').split(";")))
        features_count +=sum(len(x)*(len(x)-1)/2 for x in deep_hash_joins) #*2

interaction = parser.getboolean('config', 'interaction')
print "D=%d\ninteraction=%s"%(D, interaction)

if interaction:
	interaction_features = parser.get('config', 'interaction_features')
	hash_joins = list(list(int(z) for z in y.split(",")) for y in list(x for x in parser.get('config', 'interaction_features').split(";")))
	print "interactions=%s"%interaction_features

if parser.has_option('config', 'deep_hash_joins'): print "deep_hash_joins = %s"%parser.get('config', 'deep_hash_joins')
if parser.has_option('config', 'hash_joins'): print "hash_joins = %s"%parser.get('config', 'hash_joins')
if parser.has_option('config', 'lambada'):  print "lambada %s"% parser.getfloat('config', 'lambada') 

w=[]
n=[]
K = [k for k in range(1)]
alpha = .1   # learning rate for sgd optimization
loss_y14 = log(1. - 10**-15)

def reset_weights():
	global w,n
	del w, n
	# initialize our model, all 32 of them, again ignoring y14
	w = [[0.] * (D) if k != 13 else None for k in range(1)]
	n = [[0.] * (D) if k != 13 else None for k in range(1)]

reset_weights()

# function, generator definitions ############################################

# A. x, y generator
# INPUT:
#     path: path to train.csv or test.csv
#     label_path: (optional) path to trainLabels.csv
# YIELDS:
#     ID: id of the instance (can also acts as instance count)
#     x: a list of indices that its value is 1
#     y: (if label_path is present) label value of y1 to y33
def data(path):
    for t, line in enumerate(open(path)):
        # initialize our generator
        row = line.rstrip().split(',')
	if len(row)==23:
		row.insert(1,'0')
        if t == 0:
            # create a static x,
            # so we don't have to construct a new x for every instance
	    features_count = len(row)-2
	    if deep_hash_joins: features_count +=sum(len(x)*(len(x)-1)/2 for x in deep_hash_joins)#*2
	    if hash_joins: features_count +=len(hash_joins)#*2
	    x = [0] * (features_count)
	if row[0]=='id': continue
        # parse x
	tw=0
        for m, feat in enumerate(row):
            if m == 0:
                ID = int(feat)
            elif m == 1:
                y = [int(feat)]
            elif m == 2:
                x[tw] = abs(hash(str(m-2) + '_' + feat[6:])) % D
		date=feat[:6]
		tw += 1
            else:
                # one-hot encode everything with hash trick
                # categorical: one-hotted
                # boolean: ONE-HOTTED
                # numerical: ONE-HOTTED!
                # note, the build in hash(), although fast is not stable,
                #       i.e., same value won't always have the same hash
                #       on different machines
		x[tw] = abs(hash(str(m-2) + '_' + feat)) % D
		tw += 1
	if deep_hash_joins:			
		for i in range(len(deep_hash_joins)):
			for j in range(len(deep_hash_joins[i])-1):
				for k in range(j+1, len(deep_hash_joins[i])):
					x[tw] = abs(hash(str(tw)+"_"+row[deep_hash_joins[i][j]+2]+"_x_"+row[deep_hash_joins[i][k]+2])) % D
					tw += 1
	if hash_joins:			
		for i in range(len(hash_joins)):
			join_str=""
			for j in range(len(hash_joins[i])-1):
				join_str+=row[hash_joins[i][j]+2]+"_x_"
			join_str+=row[hash_joins[i][-1]+2]
                        x[tw] = abs(hash(str(tw)+"_"+join_str)) % D
			tw += 1

        yield (ID, date, x, y)

def train(file, validate=False):
	loss=0
	count=0
	for ID, date, x, y in data(file):
		for k in K:
			p = predict(x, w[k])
			if not validate: update(alpha, w[k], n[k], x, p, y[k])
			loss += logloss(p, y[k])  # for progressive validation
			count+=1
		loss += loss_y14  # the loss of y14, logloss is never zero
	return (loss, count)

def test(input_file, output_file):
        print 'generating submission'
        with open(output_file, 'w') as outfile:
	    outfile.write('id,click\n')
            for ID, date, x, y in data(input_file):
                for k in K:
                    p = predict(x, w[k])
                    outfile.write('%s,%.8f\n' % (ID, p))
	

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
    for i in x[:len(x)]:  # do wTx
	wTx += w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
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
    for i in x[:len(x)]:  # do wTx
        # do wTx
        # alpha / sqrt(n) is the adaptive learning rate
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1.
	n[i] += abs((p - y)  + lambada*w[i])
	w[i] -= ((p - y) * 1. + lambada*w[i]) * alpha / sqrt(n[i])
