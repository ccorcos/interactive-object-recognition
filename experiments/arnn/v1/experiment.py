#!/usr/bin/python
# coding: utf-8

# Trying an ARNN implementation using a_t and o_t to predict o_{t+1}.
# This experiment is just to see if we can learn a single die.

import theano
import theano.tensor as T
import numpy
import cPickle as pickle
from sparkprob import sparkprob
from arnn import *

print """
Running Experiment 2:
Training an ARNN on one die.
"""

with open('../datasets/one-die-optimal.pickle', 'rb') as handle:
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

arnn = ARNN(
    n = 50,
    nobs = 6,
    nact = 5,
    L1_reg = .01,
    L2_reg = .01
)

arnn.trainModel(
    observations=observations,
    actions=actions,
    learningRate=0.02,
    epochs=20
)

arnn.testModel(observations[0], actions[0])
