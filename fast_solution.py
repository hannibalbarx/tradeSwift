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
if parser.getboolean('config', 'exit'):
	print 'why you no love me? bye.'
	exit()

# TL; DR
# the main learning process start at line 122


# parameters #################################################################

train = parser.get('config', 'train_file_1')
label = parser.get('config', 'train_labels')

D = 2 ** parser.getint('config', 'D_exponent')  # number of weights use for each model, we have 32 of them
alpha = .1   # learning rate for sgd optimization

features_count = 146
deep_hash_joins=[]
hash_joins=[]
if parser.has_option('config', 'deep_hash_joins'):
	deep_hash_joins=list(list(int(z) for z in y.split(",")) for y in list(x for x in parser.get('config', 'deep_hash_joins').split(";")))
	features_count +=sum(len(x)*(len(x)-1)/2 for x in deep_hash_joins)
if parser.has_option('config', 'hash_joins'):
	hash_joins = list(list(int(z) for z in y.split(",")) for y in list(x for x in parser.get('config', 'hash_joins').split(";")))
	features_count +=len(hash_joins)

print "D_exponent = %s"%parser.get('config', 'D_exponent')
if parser.has_option('config', 'deep_hash_joins'): print "deep_hash_joins = %s"%parser.get('config', 'deep_hash_joins')
if parser.has_option('config', 'hash_joins'): print "hash_joins = %s"%parser.get('config', 'hash_joins')
print 'features count = %s'%features_count

# function, generator definitions ############################################

# A. x, y generator
# INPUT:
#     path: path to train.csv or test.csv
#     label_path: (optional) path to trainLabels.csv
# YIELDS:
#     ID: id of the instance (can also acts as instance count)
#     x: a list of indices that its value is 1
#     y: (if label_path is present) label value of y1 to y33
def data(path, label_path=None):
    for t, line in enumerate(open(path)):
        # initialize our generator
        if t == 0:
            # create a static x,
            # so we don't have to construct a new x for every instance
            ID=0
	    x = [0] * (features_count)
            if label_path:
                label = open(label_path)
                label.readline()  # we don't need the headers
            continue
        # parse x
        row = line.rstrip().split(',')
        for m, feat in enumerate(row):
            if m == 0:
                ID+= 1
            else:
                # one-hot encode everything with hash trick
                # categorical: one-hotted
                # boolean: ONE-HOTTED
                # numerical: ONE-HOTTED!
                # note, the build in hash(), although fast is not stable,
                #       i.e., same value won't always have the same hash
                #       on different machines
                x[m] = abs(hash(str(m) + '_' + feat)) % D
        tw = 145
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
    for i in x:  # do wTx
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
    for i in x:
        # alpha / sqrt(n) is the adaptive learning rate
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1.
        n[i] += abs(p - y)
        w[i] -= (p - y) * 1. * alpha / sqrt(n[i])


# training and testing #######################################################
start = datetime.now()

# a list for range(0, 33) - 13, no need to learn y14 since it is always 0
K = [k for k in range(33) if k != 13]

# initialize our model, all 32 of them, again ignoring y14
w = [[0.] * D if k != 13 else None for k in range(33)]
n = [[0.] * D if k != 13 else None for k in range(33)]

loss = 0.
loss_y14 = log(1. - 10**-15)

print 'training...'
ID=0
for ID, x, y in data(train, label):

    # get predictions and train on all labels
    for k in K:
        p = predict(x, w[k])
        update(alpha, w[k], n[k], x, p, y[k])
        loss += logloss(p, y[k])  # for progressive validation
    loss += loss_y14  # the loss of y14, logloss is never zero

    # print out progress, so that we know everything is working
    if ID % 100000 == 0:
        print('%s\tencountered: %d\tlogloss: %f' % (
            datetime.now(), ID, (loss/33.)/ID))

if parser.has_option('config', 'validation_file'):
	print 'validation...'
	loss = 0.
	ID=0
	for ID, x, y in data(parser.get('config', 'validation_file'), parser.get('config', 'validation_labels')):
	    for k in K:
		p = predict(x, w[k])
		loss += logloss(p, y[k])
	    loss += loss_y14
	if (ID):
		print('%s\tencountered: %d\t logloss: %f' % (
		    datetime.now(), ID, (loss/33.)/ID))

if parser.has_option('config', 'test_file'):
	print 'testing...'
	with open('./sontag.csv', 'w') as outfile:
	    outfile.write('id_label,pred\n')
	    for ID, x in data(parser.get('config', 'test_file')):
		for k in K:
		    p = predict(x, w[k])
		    outfile.write('%s_y%d,%s\n' % (ID, k+1, str(p)))
		    if k == 12:
			outfile.write('%s_y14,0.0\n' % ID)

print('Done, elapsed time: %s' % str(datetime.now() - start))
