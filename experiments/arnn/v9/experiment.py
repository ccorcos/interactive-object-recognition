#!/usr/bin/python
# coding: utf-8

# Trying an RNN implementation using a_t and o_t to predict o_{t+1}.
# This experiment is just to see if we can learn a single die.

import theano
import theano.tensor as T
import numpy
import cPickle as pickle
from sparkprob import sparkprob
from arnn import *
import numpy
import layer
import random


print """
Running Experiment 9:
Training an RNN on the set 1 dice. This is really just to test load and save.`
"""

# dataset = 'datasets/one-die-optimal.pickle'
dataset = 'datasets/set1-die-optimal-8.pickle'

with open('../../'+dataset, 'rb') as handle:
    samples = pickle.load(handle)

# sample = {
#     'name':die.name,
#     'note':die.note,
#     'observations': observations,
#     'actions':actions,
#     'diceProbs':diceProbs,
#     'expectedEntropies':expectedEntropies,
#     'nextProbs':nextProbs
# }

# shuffle the order... to a known one ;)
# numpy.random.set_state(0)
# numpy.random.shuffle(samples)

# nextProbs is the correct likelihood of the next feature. This will be used to
# guage how the modeled learned, but it won't be used to train the model.

values = lambda arr, key: map(lambda x: x[key], arr)
toArray = lambda x: numpy.array(x, dtype=theano.config.floatX)

actions =      toArray(values(samples, 'actions'))
observations = toArray(values(samples, 'observations'))
nextProbs =    toArray(values(samples, 'nextProbs'))


"""

                            y_{t-1}              a_t
                               |                |
                               |                |
                   observation |         action |
                               |                |
                               |                |
                    filter     ▼   transform    ▼
         k_{t-1} ----------▶  h_t ----------▶  k_t ----------▶  h_{t+1}
                                                |
                                                |
                                      predictor |
                                                |
                                                |
                                                ▼
                                               y_t

"""

trials = int(numpy.floor(actions.shape[0]/3.0))
length = 8
warmUp = 5

rnn = RNN.create(
    warmUp=warmUp,
    n_obs=6,
    n_act=5,
    n_hidden=80,
    ff_obs=[50],
    ff_filt=[50],
    ff_trans=[50],
    ff_act=[50],
    ff_pred=[50],
    L1_reg = .1,
    L2_reg = .0,
    transitionActivation='tanh',
    outputActivation='softmax'
)

lr = 0.0005
mom = 0.2
epochs = 2000

rnn.train(
    observations=observations[0:trials,0:length+1,:],
    actions=actions[0:trials,0:length,:],
    learningRate=lr,
    momentum=mom,
    epochs=epochs,
)

rnn.visualize(observations[0,0:length+1,:], actions[0,0:length,:])

testErr = rnn.test(observations[trials:,0:length+1,:], actions[trials:,0:length,:])

filename = 'rnn.pickle'
rnn.save(filename, lr=lr, mom=mom, epochs=epochs, dataset=dataset, notes="""
  80 hidden nodes with single deep 50 node forward feeds. Trained on the set1 
  8 optimal move dataset with a warmup of 5.
  """)


# Sometime later, lets load it.

rnn2, lr2, mom2, dataset2, epochs2, notes2 = RNN.load(filename)

rnn2.visualize(observations[0,0:length+1,:], actions[0,0:length,:])

testErr2 = rnn2.test(observations[trials:,0:length+1,:], actions[trials:,0:length,:])

print testErr2, "should equal", testErr