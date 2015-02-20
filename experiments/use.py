#!/usr/bin/python
# coding: utf-8

from dice.die import *
from dice.generate import *
import dice.set1 as set1

import theano
import theano.tensor as T
import numpy

from DL.models.USE import USE
from DL.optimizers.sgd import sgd
from DL.utils import *

import time
from sparkprob import sparkprob

import warnings
warnings.simplefilter("ignore")


print "Generating one trial of a dice with 20 actions"
# one trial of one die with 20 actions
# tile that one trial so that its the same training, validation and test
oneTrial = randomTrials([set1.dice[0]], 20, 1)
data = [oneTrial]*3
# dataset = randomTrials([set1[0]], 20, 3)

print "loading data to the GPU"
dataset = load_data(data, output="float32")

print "creating the USE"
o = T.tensor3('o')  # observations
a = T.tensor3('a')  # actions
inputs = [o,a]
rng = numpy.random.RandomState(int(time.time())) # random number generator

# shallow first
use = USE(
    rng=rng, 
    obs=o,
    act=a,
    n_obs=6,
    n_act=5,
    ff_obs=[20], 
    ff_filt=[20], 
    ff_trans=[20], 
    ff_act=[20], 
    ff_pred=[20],
    n_hidden=50,
    activation='relu',
    outputActivation="sigmoid"
)

# regularization
L1_reg=0.00
L2_reg=0.0001

# cost function
cost = (
    use.loss()
    + L1_reg * use.L1
    + L2_reg * use.L2_sqr
)

errors = use.errors()
params = flatten(use.params)

print "training the USE with sgdem"

sgd(dataset=dataset,
    inputs=inputs,
    cost=cost,
    params=params,
    errors=errors,
    learning_rate=0.01,
    momentum=0.2,
    n_epochs=5000,
    batch_size=100,
    patience=500,
    patience_increase=2.,
    improvement_threshold=0.9995
)


print "compiling the prediction function"

predict = theano.function(inputs=[o, a], outputs=use.output)

print "predicting the first sample of the training dataset"

obs = data[0][0][0]
act = data[0][1][0]
y = predict([obs], [act])[0]
print 'obs'.center(len(obs[0])*2) + '  |  ' + 'y'.center(len(y[0])*2) + '  |  ' + 'act'.center(len(y[0])*2)
print ''
print sparkprob(obs[0]) + '  |  ' + sparkprob([0]*6) + '  |  ' +  sparkprob(act[0])
for i in range(1, len(act)):
    print sparkprob(obs[i]) + '  |  ' + sparkprob(y[i-1]) + '  |  ' +  sparkprob(act[i])

print sparkprob(obs[len(obs)-1]) + '  |  ' + sparkprob(y[len(obs)-2]) + '  |  ' + sparkprob([0]*5)
