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
train = parser.get('config', 'train')
print "working dir = %s"%working_dir
print "training files = %s"%train
validate=None
if parser.has_option('config', 'validate'): 
	validate= parser.get('config', 'validate')
	print "validate file = %s"%(validate)
test=None
if parser.has_option('config', 'test'): 
	test = parser.get('config', 'test')
	submission = parser.get('config', 'submission_file')+'.csv'  # path of to be outputted submission file
	print "test file = %s\nsubmission file = %s"%(test, submission)

# training and testing #######################################################
start = datetime.now()

bag_of_hash.train(working_dir+train)
if validate: 
	(cur_v_loss, cur_v_count) = bag_of_hash.train(working_dir+validate, validate=True)
	print(strftime("%a %d %b %Y %H:%M:%S ")+'%s, %d, %.6f' % (validate, cur_v_count, cur_v_loss/cur_v_count))
if test:
	bag_of_hash.test(working_dir+test)

print('Done, elapsed time: %s' % str(datetime.now() - start))
