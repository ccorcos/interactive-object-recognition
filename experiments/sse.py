#!/usr/bin/python
# coding: utf-8

from dice.die import *
from dice.generate import *
import dice.set1 as set1

import theano
import theano.tensor as T
import numpy

from DL.models.SSE import SSE
from DL.optimizers.sgd import sgd
from DL.utils import *

import time
from sparkprob import sparkprob

import warnings
warnings.simplefilter("ignore")


print "Generating one trial of a dice with 20 actions"
# one trial of one die with 20 actions
# tile that one trial so that its the same training, validation and test
train_data      = supervisedData(set1.dice, 8, 1000)
validation_data = supervisedData(set1.dice, 8, 30)
test_data       = supervisedData(set1.dice, 8, 30)
data = [train_data, validation_data, test_data]


n_pred = len(set1.dice)
n_obs = 6
n_act = 5
# dataset = randomTrials([set1[0]], 20, 3)

print "loading data to the GPU"
dataset = load_data(data, output="int32")

print "creating the SSE"
o = T.tensor3('o')  # observations
a = T.tensor3('a')  # actions
t = T.imatrix('t')  # targets
inputs = [o,a,t]
rng = numpy.random.RandomState(int(time.time())) # random number generator

# shallow first
sse = SSE(
    rng=rng, 
    obs=o,
    act=a,
    n_obs=n_obs,
    n_act=n_act,
    n_pred=n_pred,
    ff_obs=[20,20], 
    ff_filt=[20,20], 
    ff_trans=[20,20], 
    ff_act=[20,20], 
    ff_pred=[20,20],
    n_hidden=50,
    activation='tanh',
    outputActivation='softmax'
)

# regularization
L1_reg=0.00
L2_reg=0.0001

# cost function
cost = (
    sse.loss(t)
    + L1_reg * sse.L1
    + L2_reg * sse.L2_sqr
)

errors = sse.errors(t)
params = flatten(sse.params)

print "training the SSE with sgdem"

sgd(dataset=dataset,
    inputs=inputs,
    cost=cost,
    params=params,
    errors=errors,
    learning_rate=0.01,
    momentum=0.2,
    n_epochs=5000,
    batch_size=200,
    patience=1000,
    patience_increase=2.,
    improvement_threshold=0.9995
)


print "compiling the prediction function"

predict = theano.function(inputs=[o, a], outputs=sse.output)

print "predicting the first sample of the training dataset"

obs = data[0][0][0]
act = data[0][1][0]
y = predict([obs], [act])[0]
print 'obs'.center(n_obs*2) + '  |  ' + 'y'.center(n_pred*2) + '  |  ' + 'act'.center(n_act*2)
print ''
print sparkprob(obs[0]) + '  |  ' + sparkprob(y[0]) + '  |  ' +  sparkprob(act[0])
for i in range(1, len(act)):
    print sparkprob(obs[i]) + '  |  ' + sparkprob(y[i]) + '  |  ' +  sparkprob(act[i])

print sparkprob(obs[len(obs)-1]) + '  |  ' + sparkprob(y[len(obs)-1]) + '  |  ' + sparkprob([0]*n_act)
