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

import bag_of_hash

from ConfigParser import SafeConfigParser
import sys

parser = SafeConfigParser()
parser.read('config.ini')
if parser.getboolean('config', 'exit'):
	print 'why you no love me? bye.'
	exit()

# parameters #################################################################

train = parser.get('config', 'train_file_1')
label = parser.get('config', 'train_labels')

D = parser.getint('config', 'D')  # number of weights use for each model, we have 32 of them
alpha = .1   # learning rate for sgd optimization

features_count = 146 +1

def trainer(hash_joins):
	# training and testing #######################################################

	# a list for range(0, 33) - 13, no need to learn y14 since it is always 0
	K = [k for k in range(33) if k != 13]

	loss = 0.
	loss_y14 = log(1. - 10**-15)

	ID2=0
	for ID, x, y in bag_of_hash.data(train, label_path=label, hash_joins=hash_joins):
	    ID2+=1
	    for k in K:
		p = bag_of_hash.predict(x, bag_of_hash.w[k])
		bag_of_hash.update(alpha, bag_of_hash.w[k], bag_of_hash.n[k], x, p, y[k])
		loss += bag_of_hash.logloss(p, y[k])  # for progressive validation
	    loss += loss_y14  # the loss of y14, logloss is never zero
	bag_of_hash.reset_weights()
	return loss, ID2

while True:
	input_str=raw_input()
	hash_joins = list(list(int(z) for z in y.split(",")) for y in list(x for x in input_str.split(";")))
	start = datetime.now()
	loss, ID2 = trainer(hash_joins)
	print('%s|%f' % (
		    input_str, (loss/33.)/ID2))
	sys.stdout.flush()
	print "done"
	sys.stdout.flush()
#print('Done, elapsed time: %s' % str(datetime.now() - start))
