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

print """
Running Experiment 2:
Training an RNN on one die.
"""

# move this into the experiment.py file
def shuffleInUnison(arr):
    rng_state = numpy.random.get_state()
    for item in arr:
        numpy.random.shuffle(item)
        numpy.random.set_state(rng_state)


with open('../../datasets/one-die-optimal.pickle', 'rb') as handle:
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
    observations.append(sample['observations'])
    nextProbs.append(sample['nextProbs'])

actions = numpy.array(actions, dtype=theano.config.floatX)
observations = numpy.array(observations, dtype=theano.config.floatX)
nextProbs = numpy.array(nextProbs, dtype=theano.config.floatX)

rnn = RNN(
    n_enc=[6, 20], 
    n_dec=[50, 20, 6], 
    n_act=[5, 20], 
    n_hid=[50, 60],
    L1_reg = .01,
    L2_reg = .01
)

shuffleInUnison([observations, actions])

rnn.trainModel(
    observations=observations,
    actions=actions,
    learningRate=0.02,
    momentum=0.1,
    epochs=100
)

rnn.testModel(observations[0], actions[0])
