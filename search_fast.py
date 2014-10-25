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

	# initialize our model, all 32 of them, again ignoring y14
	w = [[0.] * D if k != 13 else None for k in range(33)]
	n = [[0.] * D if k != 13 else None for k in range(33)]

	loss = 0.
	loss_y14 = log(1. - 10**-15)

	ID2=0
	for ID, x, y in bag_of_hash.data(D, train, label_path=label, hash_joins=hash_joins):
	    ID2+=1
	    for k in K:
		p = bag_of_hash.predict(x, w[k])
		bag_of_hash.update(alpha, w[k], n[k], x, p, y[k])
		loss += bag_of_hash.logloss(p, y[k])  # for progressive validation
	    loss += loss_y14  # the loss of y14, logloss is never zero
	del w,n
	return loss, ID2

while True:
	col1_str=raw_input()
	col1=int(col1_str)
	start = datetime.now()
	#loss, ID2 = trainer(None)
	#print('None,None,%f' % ((loss/33.)/ID2))
	for col2 in range(col1-1, -1, -1):
		loss, ID2 = trainer([[col1, col2]])
		print('%d,%d,%f' % (
			    col1, col2, (loss/33.)/ID2))
		sys.stdout.flush()
	print "done"
	sys.stdout.flush()
#print('Done, elapsed time: %s' % str(datetime.now() - start))
