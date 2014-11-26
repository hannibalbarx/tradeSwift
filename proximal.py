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

activity_bits={"14102100":"0","14102101":"0","14102102":"1","14102103":"1","14102104":"1","14102105":"1","14102106":"1","14102107":"1","14102108":"1","14102109":"1","14102110":"1","14102111":"1","14102112":"0","14102113":"1","14102114":"1","14102115":"1","14102116":"1","14102117":"1","14102118":"0","14102119":"0","14102120":"0","14102121":"0","14102122":"0","14102123":"0","14102200":"0","14102201":"0","14102202":"0","14102203":"0","14102204":"1","14102205":"1","14102206":"1","14102207":"1","14102208":"1","14102209":"1","14102210":"1","14102211":"1","14102212":"1","14102213":"1","14102214":"1","14102215":"1","14102216":"1","14102217":"1","14102218":"1","14102219":"0","14102220":"0","14102221":"0","14102222":"0","14102223":"0","14102300":"0","14102301":"0","14102302":"1","14102303":"1","14102304":"1","14102305":"1","14102306":"0","14102307":"1","14102308":"1","14102309":"1","14102310":"1","14102311":"1","14102312":"1","14102313":"1","14102314":"1","14102315":"1","14102316":"1","14102317":"1","14102318":"0","14102319":"0","14102320":"0","14102321":"0","14102322":"0","14102323":"0","14102400":"0","14102401":"0","14102402":"0","14102403":"0","14102404":"0","14102405":"0","14102406":"1","14102407":"1","14102408":"1","14102409":"0","14102410":"0","14102411":"0","14102412":"1","14102413":"1","14102414":"1","14102415":"1","14102416":"1","14102417":"1","14102418":"0","14102419":"0","14102420":"0","14102421":"0","14102422":"0","14102423":"0","14102500":"0","14102501":"0","14102502":"0","14102503":"0","14102504":"0","14102505":"0","14102506":"0","14102507":"0","14102508":"0","14102509":"1","14102510":"1","14102511":"1","14102512":"1","14102513":"1","14102514":"1","14102515":"1","14102516":"1","14102517":"1","14102518":"1","14102519":"0","14102520":"0","14102521":"0","14102522":"0","14102523":"0","14102600":"0","14102601":"0","14102602":"0","14102603":"0","14102604":"1","14102605":"1","14102606":"1","14102607":"1","14102608":"1","14102609":"1","14102610":"1","14102611":"1","14102612":"1","14102613":"1","14102614":"1","14102615":"1","14102616":"1","14102617":"1","14102618":"0","14102619":"0","14102620":"0","14102621":"0","14102622":"0","14102623":"0","14102700":"0","14102701":"0","14102702":"0","14102703":"0","14102704":"0","14102705":"1","14102706":"1","14102707":"1","14102708":"1","14102709":"0","14102710":"0","14102711":"0","14102712":"0","14102713":"1","14102714":"0","14102715":"0","14102716":"1","14102717":"1","14102718":"1","14102719":"0","14102720":"0","14102721":"0","14102722":"0","14102723":"0","14102800":"0","14102801":"0","14102802":"0","14102803":"0","14102804":"0","14102805":"1","14102806":"1","14102807":"1","14102808":"1","14102809":"1","14102810":"1","14102811":"1","14102812":"1","14102813":"1","14102814":"1","14102815":"1","14102816":"1","14102817":"1","14102818":"1","14102819":"0","14102820":"0","14102821":"0","14102822":"0","14102823":"0","14102900":"0","14102901":"0","14102902":"0","14102903":"1","14102904":"0","14102905":"0","14102906":"0","14102907":"1","14102908":"1","14102909":"0","14102910":"1","14102911":"0","14102912":"0","14102913":"1","14102914":"1","14102915":"1","14102916":"0","14102917":"0","14102918":"0","14102919":"0","14102920":"0","14102921":"0","14102922":"0","14102923":"0","14103000":"0","14103001":"0","14103002":"0","14103003":"0","14103004":"1","14103005":"1","14103006":"1","14103007":"0","14103008":"0","14103009":"1","14103010":"1","14103011":"1","14103012":"1","14103013":"1","14103014":"1","14103015":"1","14103016":"1","14103017":"1","14103018":"1","14103019":"0","14103020":"0","14103021":"0","14103022":"0","14103023":"0"}

##############################################################################
# parameters #################################################################
##############################################################################

print strftime("%a %d %b %Y %H:%M:%S")+" wassup"
from ConfigParser import SafeConfigParser
parser = SafeConfigParser()
parser.read('config.ini')

# A, paths
train = parser.get('config', 'train_file_1')
test=validation=None
print "train file = %s"%train
if parser.has_option('config', 'validation_file'): 
	validation = parser.get('config', 'validation_file')
	print "validation file = %s"%validation
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

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
	    for i in range(len(hash_joins)):
		join_str=""
		for j in range(len(hash_joins[i])-1):
			join_str+=str(hash_joins[i][j])+"_"+str(x[hash_joins[i][j]])+"_"
		join_str+=str(hash_joins[i][-1])+"_"+str(x[hash_joins[i][-1]])
		yield abs(hash(str(i)+"_"+str(j)+"_"+join_str)) % D
            '''L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D'''

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


	activity_bit = activity_bits[row[0]]
	row.append(activity_bit)

        # extract date
        date = int(row[0][4:6])

        # turn hour really into hour, it was originally YYMMDDHH
        row[0] = row[0][6:]

        # build x
        x = []
        for key in range(len(row)):
            value = row[key]

            # one-hot encode everything with hash trick
            index = abs(hash(str(key) + '_' + value)) % D
            x.append(index)

        yield t, date, ID, x, y


##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()

# initialize ourselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)

# start training
e=0
while (e<epoch or epoch==0):
    loss = 0.
    count = 0

    for t, date, ID, x, y in data(train, D):  # data is a generator
        #    t: just a instance counter
        # date: you know what this is
        #   ID: id provided in original data
        #    x: features
        #    y: label (click)

        # step 1, get prediction from learner
        p = learner.predict(x)

        learner.update(x, p, y)
	
    print('epoch %d finished, elapsed time: %s'%(e, str(datetime.now() - start)))

    if validation:
	v_loss=0
	v_count=0
	for t, date, ID, x, y in data(validation, D):
		p = learner.predict(x)
		v_count+=1
		v_loss+=logloss(p, y)
	print('validation file logloss: %f' % (v_loss/v_count))
    
    if test: 
	with open(parser.get('config', 'submission_file')+'.'+strftime("%d%b%H%M")+'.'+str(e)+'.csv', 'w') as outfile:
	    outfile.write('id,click\n')
	    for t, date, ID, x, y in data(test, D):
		p = learner.predict(x)
		outfile.write('%s,%.8f\n' % (ID, p))
    e+=1


##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################

print strftime("%a %d %b %Y %H:%M:%S")+" imma bounce"
