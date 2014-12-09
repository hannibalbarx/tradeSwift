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

# training and testing #######################################################
start = datetime.now()

if test:
	for current_training_file_index in range(len(training_files)):
		bag_of_hash.train(working_dir+training_files[current_training_file_index])
	bag_of_hash.test(working_dir+parser.get('config', 'test_file'), 'fast_'+parser.get('config', 'submission_file')+'.'+strftime("%d%b%H%M")+'.csv')
else:	
	file_count=len(training_files)
	if file_count<2:
		print "too few files for cv fold"
		exit()
	v_loss=0
	v_count=0
	for validation_file_index in range(len(training_files)):
		for current_training_file_index in range(len(training_files)):
			if current_training_file_index == validation_file_index and len(training_files)>1: continue
			bag_of_hash.train(working_dir+training_files[current_training_file_index])
		(cur_v_loss, cur_v_count) = bag_of_hash.train(working_dir+training_files[validation_file_index], validate=True)
		v_loss+= cur_v_loss; v_count+=cur_v_count
		print(strftime("%a %d %b %Y %H:%M:%S ")+'%s, %d, %.6f, %d, %6f' % (training_files[validation_file_index], cur_v_count, cur_v_loss/cur_v_count,v_count, v_loss/v_count))
		if len(training_files)-validation_file_index >1: bag_of_hash.reset_weights()

print('Done, elapsed time: %s' % str(datetime.now() - start))
