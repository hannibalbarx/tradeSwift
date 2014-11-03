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

'''while True:
	input_str=raw_input()
	hash_joins = list(list(int(z) for z in y.split(",")) for y in list(x for x in input_str.split(";")))
	start = datetime.now()
	loss, ID2 = trainer(hash_joins)
	print('%s|%f' % (
		    input_str, (loss/33.)/ID2))
	sys.stdout.flush()
	print "done"
	sys.stdout.flush()'''
#print('Done, elapsed time: %s' % str(datetime.now() - start))
input_str=""
a="128,91;91,31;91,30;140,91;142,91;126,91;141,91;141,91;130,91;127,91;129,91;92,91;105,91;116,91;102,91;104,91;93,91;91,42;91,44;91,56;91,33;91,63;91,32;115,91;117,91;101,91;91,74;91,62;91,86;91,72;91,26;91,24;91,45;91,10;91,25;91,14;91,2;91,1;91,13;91,11;91,75;103,91;91,55;91,41;91,57;91,12;91,43;91,85;91,87;91,71;91,73;142,131;140,131;131,126;142,133;140,133;133,126;133,128;133,17;133,10;133,130;133,26;133,24;133,117;133,101;133,115;94,31;133,31;133,23;133,22;133,41;94,30;133,12;133,57;133,55;141,133;133,105;133,1;133,2;133,30;133,113;133,14;133,13;133,11;133,92;133,103;133,25;133,71;133,85;133,87;133,131;138,133;133,114;131,10;133,104;131,26;133,53;133,102;95,26;95,24"

for i in a.split(";"):
	if input_str: input_str+=";"
	input_str+=i
	hash_joins = list(list(int(z) for z in y.split(",")) for y in list(x for x in input_str.split(";")))
	loss, ID2 = trainer(hash_joins)
	print('%s|%f' % (
		input_str, (loss/33.)/ID2))
