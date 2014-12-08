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
import bag_of_hash

# TL; DR
# the main learning process start at line 122


# parameters #################################################################

working_dir = parser.get('config', 'working_dir')
training_files = parser.get('config', 'training_files').split(";")
print "working dir = %s"%working_dir
print "training files = %s"%training_files

test=None
if parser.has_option('config', 'test_file'): 
	test = parser.get('config', 'test_file')
	submission = parser.get('config', 'submission_file')+'.csv'  # path of to be outputted submission file
	print "test file = %s\nsubmission file = %s"%(test, submission)

alpha = .1   # learning rate for sgd optimization

features_count = 146
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

print "D= %s"%parser.get('config', 'D')
if parser.has_option('config', 'deep_hash_joins'): print "deep_hash_joins = %s"%parser.get('config', 'deep_hash_joins')
if parser.has_option('config', 'hash_joins'): print "hash_joins = %s"%parser.get('config', 'hash_joins')
if parser.has_option('config', 'lambada'):  print "lambada %s"% parser.getfloat('config', 'lambada') 
#print 'features count = %s'%features_count

# training and testing #######################################################
start = datetime.now()

# a list for range(0, 33) - 13, no need to learn y14 since it is always 0
K = [k for k in range(1)]

loss_y14 = log(1. - 10**-15)

v_loss=0
v_count=0
for validation_file_index in range(len(training_files)):

	loss = 0.
	count = 0
	
	for current_training_file_index in range(len(training_files)):

		if current_training_file_index == validation_file_index and len(training_files)>1:
			continue

		for ID, date, x, y in bag_of_hash.data(working_dir+training_files[current_training_file_index], deep_hash_joins, hash_joins):
			for k in K:
				p = bag_of_hash.predict(x, bag_of_hash.w[k])
				bag_of_hash.update(alpha, bag_of_hash.w[k], bag_of_hash.n[k], x, p, y[k])
				loss += bag_of_hash.logloss(p, y[k])  # for progressive validation
			loss += loss_y14  # the loss of y14, logloss is never zero
	if len(training_files)>1:
		cur_v_loss=0
		cur_v_count=0
		for ID, date, x, y in bag_of_hash.data(working_dir+training_files[validation_file_index], deep_hash_joins, hash_joins):
			v_count+=1; cur_v_count+=1
			for k in K:
				p = bag_of_hash.predict(x, bag_of_hash.w[k])
				l=bag_of_hash.logloss(p, y[k]); v_loss+= l; cur_v_loss+= l;
			v_loss+= loss_y14; cur_v_loss+= loss_y14  # the loss of y14, logloss is never zero
		print(strftime("%a %d %b %Y %H:%M:%S ")+'%s, %d, %.6f, %d, %6f' % (training_files[validation_file_index], cur_v_count, cur_v_loss/cur_v_count,v_count, v_loss/v_count))
		if len(training_files)-validation_file_index >1: bag_of_hash.reset_weights()

if test:
        print 'generating submission'
        with open('fast_'+parser.get('config', 'submission_file')+'.'+strftime("%d%b%H%M")+'.csv', 'w') as outfile:
	    outfile.write('id,click\n')
            for ID, date, x, y in bag_of_hash.data(working_dir+parser.get('config', 'test_file'), deep_hash_joins, hash_joins):
                for k in K:
                    p = bag_of_hash.predict(x, bag_of_hash.w[k])
                    outfile.write('%s,%.8f\n' % (ID, p))


print('Done, elapsed time: %s' % str(datetime.now() - start))
