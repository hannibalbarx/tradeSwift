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
train = parser.get('config', 'train').split(',')
print "working dir = %s"%working_dir
print "training files = %s"%train
validate=None
if parser.has_option('config', 'validate'): 
	validate= parser.get('config', 'validate').split(',')
	print "validate files = %s"%(validate)
test=None
if parser.has_option('config', 'test'): 
	test = parser.get('config', 'test')
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

d={}
d['14102100']=119006
d['14102101']=137442
d['14102102']=207471
d['14102103']=193355
d['14102104']=264711
d['14102105']=273500
d['14102106']=239720
d['14102107']=209311
d['14102108']=207244
d['14102109']=230917
d['14102110']=200028
d['14102111']=175666
d['14102112']=143620
d['14102113']=190481
d['14102114']=174531
d['14102115']=176156
d['14102116']=171869
d['14102117']=171933
d['14102118']=152365
d['14102119']=119775
d['14102120']=112017
d['14102121']=89501
d['14102122']=88436
d['14102123']=73940
d['14102200']=78006
d['14102201']=100611
d['14102202']=102844
d['14102203']=137006
d['14102204']=200948
d['14102205']=286697
d['14102206']=288819
d['14102207']=212323
d['14102208']=322803
d['14102209']=447783
d['14102210']=438270
d['14102211']=386757
d['14102212']=408650
d['14102213']=323480
d['14102214']=185611
d['14102215']=185381
d['14102216']=184316
d['14102217']=196912
d['14102218']=205756
d['14102219']=173603
d['14102220']=144375
d['14102221']=123826
d['14102222']=109372
d['14102223']=92977
d['14102300']=89341
d['14102301']=102258
d['14102302']=147980
d['14102303']=175094
d['14102304']=273173
d['14102305']=234524
d['14102306']=138628
d['14102307']=160004
d['14102308']=232430
d['14102309']=185193
d['14102310']=162806
d['14102311']=161494
d['14102312']=152413
d['14102313']=162221
d['14102314']=173837
d['14102315']=246297
d['14102316']=178192
d['14102317']=158332
d['14102318']=157759
d['14102319']=132883
d['14102320']=133045
d['14102321']=116964
d['14102322']=105652
d['14102323']=90232
d['14102400']=82137
d['14102401']=94622
d['14102402']=125003
d['14102403']=123845
d['14102404']=161530
d['14102405']=168185
d['14102406']=169842
d['14102407']=181603
d['14102408']=173242
d['14102409']=167755
d['14102410']=146693
d['14102411']=135783
d['14102412']=188407
d['14102413']=189058
d['14102414']=217145
d['14102415']=206683
d['14102416']=252154
d['14102417']=262579
d['14102418']=170884
d['14102419']=34163
d['14102420']=23124
d['14102421']=20367
d['14102422']=14876
d['14102423']=25622
d['14102500']=73446
d['14102501']=80637
d['14102502']=38400
d['14102503']=67443
d['14102504']=76816
d['14102505']=117207
d['14102506']=51726
d['14102507']=136398
d['14102508']=149460
d['14102509']=180060
d['14102510']=182311
d['14102511']=195507
d['14102512']=205914
d['14102513']=238071
d['14102514']=242500
d['14102515']=238427
d['14102516']=210066
d['14102517']=205320
d['14102518']=191373
d['14102519']=122147
d['14102520']=100957
d['14102521']=92722
d['14102522']=89708
d['14102523']=76506
d['14102600']=72185
d['14102601']=80863
d['14102602']=100821
d['14102603']=112022
d['14102604']=150184
d['14102605']=171123
d['14102606']=179145
d['14102607']=200176
d['14102608']=188781
d['14102609']=201902
d['14102610']=219830
d['14102611']=239312
d['14102612']=237815
d['14102613']=236291
d['14102614']=233840
d['14102615']=206823
d['14102616']=201409
d['14102617']=183154
d['14102618']=153815
d['14102619']=116956
d['14102620']=101676
d['14102621']=82130
d['14102622']=88869
d['14102623']=76770
d['14102700']=62254
d['14102701']=75382
d['14102702']=85828
d['14102703']=120546
d['14102704']=136684
d['14102705']=168542
d['14102706']=172517
d['14102707']=162343
d['14102708']=163554
d['14102709']=152066
d['14102710']=101249
d['14102711']=118272
d['14102712']=166229
d['14102713']=173052
d['14102714']=105178
d['14102715']=104242
d['14102716']=179656
d['14102717']=234152
d['14102718']=202275
d['14102719']=157962
d['14102720']=107591
d['14102721']=93247
d['14102722']=94982
d['14102723']=87207
d['14102800']=82520
d['14102801']=102748
d['14102802']=174692
d['14102803']=120099
d['14102804']=140573
d['14102805']=185896
d['14102806']=187063
d['14102807']=210758
d['14102808']=291763
d['14102809']=328576
d['14102810']=258656
d['14102811']=257891
d['14102812']=272124
d['14102813']=432308
d['14102814']=387453
d['14102815']=282315
d['14102816']=283491
d['14102817']=280835
d['14102818']=232346
d['14102819']=201072
d['14102820']=170081
d['14102821']=166216
d['14102822']=124195
d['14102823']=113551
d['14102900']=98950
d['14102901']=118914
d['14102902']=141174
d['14102903']=228057
d['14102904']=160923
d['14102905']=131740
d['14102906']=142879
d['14102907']=228905
d['14102908']=201567
d['14102909']=186264
d['14102910']=224107
d['14102911']=161531
d['14102912']=192389
d['14102913']=193327
d['14102914']=224237
d['14102915']=202404
d['14102916']=177356
d['14102917']=158732
d['14102918']=132239
d['14102919']=125421
d['14102920']=117017
d['14102921']=95752
d['14102922']=90365
d['14102923']=98358
d['14103000']=87333
d['14103001']=91307
d['14103002']=98459
d['14103003']=121534
d['14103004']=347806
d['14103005']=244765
d['14103006']=192404
d['14103007']=155891
d['14103008']=165420
d['14103009']=195885
d['14103010']=215813
d['14103011']=219810
d['14103012']=249022
d['14103013']=250441
d['14103014']=259187
d['14103015']=232694
d['14103016']=211916
d['14103017']=177578
d['14103018']=159570
d['14103019']=134243
d['14103020']=110764
d['14103021']=112238
d['14103022']=101250
d['14103023']=83608

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

	c=d[row[0]]
	if c<75000:
		cfield="1"
	elif c<75000 +67000:
		cfield="2"
	elif c<75000 +67000+67000:
		cfield="3"
	elif c<75000 +67000+67000+67000:
		cfield="4"
	else:
		cfield="5"
	
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

	index = abs(hash("count" + '_' + cfield)) % D
	x.append(index)
	
        # now yield interactions (if applicable)
        if interaction:
	    for i in range(len(hash_joins)):
		join_str=""
		for j in range(len(hash_joins[i])-1):
			join_str+=str(hash_joins[i][j])+"_"+str(row[hash_joins[i][j]])+"_"
		join_str+=str(hash_joins[i][-1])+"_"+str(row[hash_joins[i][-1]])
		index = abs(hash(join_str)) % D
		x.append(index)
        yield t, date, ID, x, y


##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()

if test: 
	learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)
	e=0
	while (e<epoch):
		cur_v_loss=0
		cur_v_count=0
		for tr in train:
			for t, date, ID, x, y in data(working_dir+tr, D):  # data is a generator
				p = learner.predict(x)
				learner.update(x, p, y)
				cur_v_count+=1
				cur_v_loss+=logloss(p, y) 
		print(strftime("%a %d %b %Y %H:%M:%S ")+'epoch %d, %d, %.6f' % (e, cur_v_count, cur_v_loss/cur_v_count)),
		e+=1
		learner.alpha=.85*learner.alpha
	with open(parser.get('config', 'submission_file')+'.'+strftime("%d%b%H%M")+'.csv', 'w') as outfile:
		outfile.write('id,click\n')
		for t, date, ID, x, y in data(working_dir+test, D):
			p = learner.predict(x)
			outfile.write('%s,%.8f\n' % (ID, p))
	sys.exit()
	
for v in validate: 
	learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)
	e=0
	while (e<epoch):
		cur_v_loss=0
		cur_v_count=0
		for tr in train:
			if tr==v: continue
			for t, date, ID, x, y in data(working_dir+tr, D):  # data is a generator
				p = learner.predict(x)
				learner.update(x, p, y)
				cur_v_count+=1
				cur_v_loss+=logloss(p, y) 
		print(strftime("%a %d %b %Y %H:%M:%S ")+'epoch %d, %d, %.6f' % (e, cur_v_count, cur_v_loss/cur_v_count)),
		cur_v_loss=0
		cur_v_count=0
		for t, date, ID, x, y in data(working_dir+v, D):
			p = learner.predict(x)
			cur_v_count+=1
			cur_v_loss+=logloss(p, y) 
		print(', %s, %d, %.6f' % (v, cur_v_count, cur_v_loss/cur_v_count))
		e+=1
		learner.alpha=.85*learner.alpha


##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################

print strftime("%a %d %b %Y %H:%M:%S")+" imma bounce"
