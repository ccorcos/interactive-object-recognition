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
Running Experiment 7:
Training an RNN on one die.
"""

# move this into the experiment.py file
def shuffleInUnison(arr):
    rng_state = numpy.random.get_state()
    for item in arr:
        numpy.random.shuffle(item)
        numpy.random.set_state(rng_state)


with open('../../datasets/one-die-optimal.pickle', 'rb') as handle:
# with open('../../datasets/set1-die-optimal-8.pickle', 'rb') as handle:
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

# nextProbs is the correct likelihood of the next feature. This will be used to
# guage how the modeled learned, but it won't be used to train the model.

actions = []
observations = []
nextProbs = []
for sample in samples:
    actions.append(sample['actions'])
    # obs = sample['observations']
    # map(lambda x: 0.1 if x==0 else 0.9, obs)
    observations.append(sample['observations'])
    nextProbs.append(sample['nextProbs'])

actions = numpy.array(actions, dtype=theano.config.floatX)
observations = numpy.array(observations, dtype=theano.config.floatX)
nextProbs = numpy.array(nextProbs, dtype=theano.config.floatX)

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

trials = 30
length = 12
warmUp = 5

rnn = RNN(
    warmUp=6,
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
    transitionActivation=layer.tanh,
    outputActivation=layer.softmax
)

# shuffleInUnison([observations, actions])


rnn.trainModel(
    observations=observations[0:trials,0:length+1,:],
    actions=actions[0:trials,0:length,:],
    learningRate=0.1,
    momentum=0.1,
    epochs=100,
)
rnn.testModel(observations[0,0:length+1,:], actions[0,0:length,:])
