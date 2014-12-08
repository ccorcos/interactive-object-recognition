#!/usr/bin/python
# coding: utf-8

# Trying a RNN implementation using a_t and o_t to predict o_{t+1} through a
# recurrent hidden layer.

# This experiment is just to see if we can learn a single die. This should be
# trivially easy because every o_t and a_t have a direct correspondence to the
# o_{t+1}.

import theano
import theano.tensor as T
import numpy
import cPickle as pickle
from sparkprob import sparkprob
from rnn import *

print """
Running Experiment 1:
Training a RNN on one die.
"""

# with open('../../datasets/one-die-optimal.pickle', 'rb') as handle:
with open('../../datasets/set1-not-optimal-50.pickle', 'rb') as handle:
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

inputs = []
targets = []
nextProbs = []
for sample in samples:
    i = []
    o = []
    p = []
    actions = sample['actions']
    observations = sample['observations']
    nextProb = sample['nextProbs']
    for t in range(len(observations)-1):
        i.append(actions[t]+observations[t])
        o.append(observations[t+1])
        p.append(nextProb[t])
    inputs.append(i)
    targets.append(o)
    nextProbs.append(p)

inputs = numpy.array(inputs, dtype=theano.config.floatX)
targets = numpy.array(targets, dtype=theano.config.floatX)
nextProbs = numpy.array(nextProbs, dtype=theano.config.floatX)



rnn = RNN(
    n = 50,
    nin = 11,
    nout = 6,
    L1_reg = .01,
    L2_reg = .01
)

rnn.trainModel(
    inputs=inputs,
    targets=nextProbs,
    learningRate=0.01,
    momentum=0.6,
    epochs=100
)

rnn.testModel(inputs[0], nextProbs[0])
