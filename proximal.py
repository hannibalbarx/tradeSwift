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
import sys

from csv import DictReader
from math import exp, log, sqrt


# TL; DR, the main training process starts on line: 250,
# you may want to start reading the code from there


##############################################################################
# parameters #################################################################
##############################################################################

print strftime("%a %d %b %Y %H:%M:%S ")+"wassup"
from ConfigParser import SafeConfigParser
parser = SafeConfigParser()
parser.read('config.ini')

working_dir = parser.get('config', 'working_dir')
training_files = parser.get('config', 'training_files').split(";")
print "working dir = %s"%working_dir
print "training files = %s"%training_files

test=None
if parser.has_option('config', 'test_file'): 
	test = parser.get('config', 'test_file')
	submission = parser.get('config', 'submission_file')+'.csv'  # path of to be outputted submission file
	print "test file = %s\nsubmission file = %s"%(test, submission)

# B, model
alpha = parser.getfloat('config', 'alpha')  # learning rate
beta = parser.getfloat('config', 'beta')   # smoothing parameter for adaptive learning rate
L1 = parser.getfloat('config', 'L1')      # L1 regularization, larger value means more regularized
L2 = parser.getfloat('config', 'L2')     # L2 regularization, larger value means more regularized
print "alpha=%f\nbeta=%f\nL1=%f\nL2=%f"%(alpha, beta, L1, L2)

# C, feature/hash trick
D = parser.getint('config', 'D') 
interaction = parser.getboolean('config', 'interaction')
print "D=%d\ninteraction=%s"%(D, interaction)

if interaction:
	interaction_features = parser.get('config', 'interaction_features')
	hash_joins = list(list(int(z) for z in y.split(",")) for y in list(x for x in parser.get('config', 'interaction_features').split(";")))
	print "interactions=%s"%interaction_features
kill=None
if parser.has_option('config', 'kill') :
	kill=map(int,parser.get('config', 'kill').split(","))
	print "kill = %s"%kill
choose=None
if parser.has_option('config', 'choose') :
	choose=map(int,parser.get('config', 'choose').split(","))
	print "choose = %s"%choose

# D, training/validation
holdafter = None
if parser.has_option('config', 'holdafter') :
	holdafter = parser.getint('config', 'holdafter')
	print "holdafter=%d"%holdafter
holdout = None
if parser.has_option('config', 'holdout'):
	holdout = parser.getint('config', 'holdout') 
	print "holdout=%d"%holdout
epoch = parser.getint('config', 'epochs')
if not epoch:
	print "infinite epochs"
else:
	print "epochs=%d"%epoch

##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


def data(path, D):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''

    for t, row in enumerate((open(path))):
	row = row.split(",")
	if row[0]=="id": continue
        # process id
        ID = row[0]
        del row[0]

        # process clicks
        y = 0.
        if len(row)==23:
            if row[0] == '1':
                y = 1.
            del row[0]


        # extract date
        date = int(row[0][4:6])

        # turn hour really into hour, it was originally YYMMDDHH
        row[0] = row[0][6:]

        # build x
        x = []
        for key in range(len(row)):
		if (not kill or key not in kill) and (not choose or key in choose):
			    value = row[key]
			    # one-hot encode everything with hash trick
			    index = abs(hash(str(key) + '_' + value)) % D
			    x.append(index)
        # now yield interactions (if applicable)
        if interaction:
	    for i in range(len(hash_joins)):
		join_str=""
		for j in range(len(hash_joins[i])-1):
			join_str+=str(hash_joins[i][j])+"_"+str(row[hash_joins[i][j]])+"_"
		join_str+=str(hash_joins[i][-1])+"_"+str(row[hash_joins[i][-1]])
		index = abs(hash(str(i)+"_"+str(j)+"_"+join_str)) % D
		x.append(index)
        yield t, date, ID, x, y


##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()

v_loss=0
v_count=0
for validation_file_index in range(len(training_files)):

	# initialize ourselves a learner
	learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)

	# start training
	e=0
	while (e<epoch or epoch==0):
		loss = 0.
		count = 0
	    
		for current_training_file_index in range(len(training_files)):

		    if current_training_file_index == validation_file_index: continue
		    
		    for t, date, ID, x, y in data(working_dir+training_files[current_training_file_index] , D):  # data is a generator
			#    t: just a instance counter
			# date: you know what this is
			#   ID: id provided in original data
			#    x: features
			#    y: label (click)

			# step 1, get prediction from learner
			p = learner.predict(x)

			learner.update(x, p, y)
			
		#print(strftime("%a %d %b %Y %H:%M:%S ")+'epoch %d finished, elapsed time: %s'%(e, str(datetime.now() - start)))
		e+=1

	cur_v_loss=0
	cur_v_count=0
	for t, date, ID, x, y in data(working_dir+training_files[validation_file_index] , D):
		p = learner.predict(x)
		v_count+=1; cur_v_count+=1
		l=logloss(p, y); v_loss+=l; cur_v_loss+=l
	print(strftime("%a %d %b %Y %H:%M:%S ")+'%s, %d, %.6f, %d, %6f' % (training_files[validation_file_index], cur_v_count, cur_v_loss/cur_v_count,v_count, v_loss/v_count))
	if not test: del learner

if test: 
	with open(parser.get('config', 'submission_file')+'.'+strftime("%d%b%H%M")+'.'+str(e)+'.csv', 'w') as outfile:
	    outfile.write('id,click\n')
	    for t, date, ID, x, y in data(test, D):
		p = learner.predict(x)
		outfile.write('%s,%.8f\n' % (ID, p))

##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################

print strftime("%a %d %b %Y %H:%M:%S")+" imma bounce"
