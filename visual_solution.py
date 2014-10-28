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
import bag_of_hash

# parameters #################################################################

train = parser.get('config', 'train_file_1')
label = parser.get('config', 'train_labels')

alpha = .1   # learning rate for sgd optimization

features_count = 146
deep_hash_joins=[]
hash_joins=[]
if parser.has_option('config', 'deep_hash_joins'):
        deep_hash_joins=list(list(int(z) for z in y.split(",")) for y in list(x for x in parser.get('config', 'deep_hash_joins').split(";")))
        features_count +=sum(len(x)*(len(x)-1)/2 for x in deep_hash_joins) #*2

if parser.has_option('config', 'hash_joins'):
        hash_joins = list(list(int(z) for z in y.split(",")) for y in list(x for x in parser.get('config', 'hash_joins').split(";")))
        features_count +=len(hash_joins) #*2
#if bag_of_hash.non_hash: features_count +=len(bag_of_hash.non_hash)
non_hash=list(int(z) for z in parser.get('config', 'non_hash').split(","))

print "D= %s"%parser.get('config', 'D')
if parser.has_option('config', 'deep_hash_joins'): print "deep_hash_joins = %s"%parser.get('config', 'deep_hash_joins')
if parser.has_option('config', 'hash_joins'): print "hash_joins = %s"%parser.get('config', 'hash_joins')
if parser.has_option('config', 'non_hash'): print "non_hash = %s"%parser.get('config', 'non_hash')
print 'features count = %s'%features_count

def update(alpha, w, n, x, p, y):
    global D,non_hash
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

def data(path, label_path=None, deep_hash_joins=None, hash_joins=None):
    for t, line in enumerate(open(path)):
        # initialize our generator
        if t == 0:
            # create a static x,
            # so we don't have to construct a new x for every instance
	    features_count = 146
	    if deep_hash_joins: features_count +=sum(len(x)*(len(x)-1)/2 for x in deep_hash_joins)#*2
	    if hash_joins: features_count +=len(hash_joins)#*2
	    #if non_hash: features_count +=len(non_hash)
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
		if m in non_hash: floats+=[float(feat)]
		else: x[m] = abs(hash(str(m) + '_' + feat)) % D
        tw = 145 - len(floats)
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


# training and testing #######################################################
start = datetime.now()

# a list for range(0, 33) - 13, no need to learn y14 since it is always 0
K = [k for k in range(33) if k != 13]

# initialize our model, all 32 of them, again ignoring y14
w = [[0.] * (D+len(bag_of_hash.non_hash)) if k != 13 else None for k in range(33)]
n = [[0.] * (D+len(bag_of_hash.non_hash)) if k != 13 else None for k in range(33)]

loss = 0.
loss_y14 = log(1. - 10**-15)

print 'training...'
ID2=0
for ID, x, y in data(train, label, deep_hash_joins, hash_joins):
    ID2+=1
    for k in K:
        p = bag_of_hash.predict(x, w[k])
        update(alpha, w[k], n[k], x, p, y[k])
        loss += bag_of_hash.logloss(p, y[k])  # for progressive validation
    loss += loss_y14  # the loss of y14, logloss is never zero

    # print out progress, so that we know everything is working
    if ID % 100000 == 0:
        print('%s\tencountered: %d\tlogloss: %f' % (
            datetime.now(), ID2, (loss/33.)/ID2))

