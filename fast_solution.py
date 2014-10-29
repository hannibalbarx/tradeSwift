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

D = parser.getint('config', 'D')  # number of weights use for each model, we have 32 of them
import bag_of_hash

# TL; DR
# the main learning process start at line 122


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
if bag_of_hash.non_hash and bag_of_hash.hash_non_hash: features_count +=len(bag_of_hash.non_hash)

print "D= %s"%parser.get('config', 'D')
if parser.has_option('config', 'deep_hash_joins'): print "deep_hash_joins = %s"%parser.get('config', 'deep_hash_joins')
if parser.has_option('config', 'hash_joins'): print "hash_joins = %s"%parser.get('config', 'hash_joins')
if bag_of_hash.non_hash: print "non_hash = %s"%bag_of_hash.non_hash
if bag_of_hash.hash_non_hash: print "hash_non_hash = true"
if parser.has_option('config', 'lambada'):  print "lambada %s"% parser.getfloat('config', 'lambada') 
print 'features count = %s'%features_count

# training and testing #######################################################
start = datetime.now()

# a list for range(0, 33) - 13, no need to learn y14 since it is always 0
K = [k for k in range(33) if k != 13]

loss = 0.
loss_y14 = log(1. - 10**-15)

print 'training...'
ID2=0
for ID, x, y in bag_of_hash.data(train, label, deep_hash_joins, hash_joins):
    ID2+=1
    for k in K:
        p = bag_of_hash.predict(x, bag_of_hash.w[k])
        bag_of_hash.update(alpha, bag_of_hash.w[k], bag_of_hash.n[k], x, p, y[k])
        loss += bag_of_hash.logloss(p, y[k])  # for progressive validation
    loss += loss_y14  # the loss of y14, logloss is never zero

    # print out progress, so that we know everything is working
    if ID % 100000 == 0:
        print('%s\tencountered: %d\tlogloss: %f' % (
            datetime.now(), ID2, (loss/33.)/ID2))

if parser.has_option('config', 'validation_file'):
        print 'validation...'
        loss3 = 0.
        ID3=0
        for ID, x, y in bag_of_hash.data(parser.get('config', 'validation_file'), parser.get('config', 'validation_labels'), deep_hash_joins, hash_joins):
            ID3+=1
            for k in K:
                p = bag_of_hash.predict(x, w[k])
                loss3 += bag_of_hash.logloss(p, y[k])
            loss3 += loss_y14
        if (ID3):
                print('%s\tencountered: %d\tlogloss: %f' % (
                    datetime.now(), ID3, (loss3/33.)/ID3))

if parser.has_option('config', 'test_file'):
        print 'testing...'
        with open('./sontag.csv', 'w') as outfile:
            outfile.write('id_label,pred\n')
            for ID, x in bag_of_hash.data(parser.get('config', 'test_file'), deep_hash_joins, hash_joins):
                for k in K:
                    p = bag_of_hash.predict(x, w[k])
                    outfile.write('%s_y%d,%s\n' % (ID, k+1, str(p)))
                    if k == 12:
                        outfile.write('%s_y14,0.0\n' % ID)

print('Done, elapsed time: %s' % str(datetime.now() - start))
