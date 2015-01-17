#!/usr/bin/python
# coding: utf-8

# Trying a RNN implementation using a_t and o_t to predict o_{t+1} through a
# recurrent hidden layer.

# This experiment is just to see if we can learn a single die. This should be
# trivially easy because every o_t and a_t have a direct correspondence to the
from sparkprob import sparkprob
from rnn import *
from utils import *

print """
Running Experiment 7:
Training a RNN on set1 with deep inputs, outputs, and transitions.
"""

trainingDataset = 'set1-not-optimal-50.pickle'

inputs, targets, nextProbs = getData(trainingDataset)

relu = lambda x: x * (x > 0)
cappedrelu =  lambda x: T.minimum(x * (x > 0), 6)

rnn = RNN(
    n = 50,
    m = 30,
    nin = 11,
    min = 40,
    nout = 6,
    mout = 40,
    L1_reg = 0,
    L2_reg = 0,
    transitionActivation=relu,
    outputActivation=T.nnet.sigmoid
)

rnn.trainModel(
    inputs=inputs,
    targets=targets,
    learningRate=0.01,
    momentum=0.1,
    epochs=300
)

rnn.testModel(inputs[0], targets[0])


print "training dataset:", trainingDataset
print "training error:", rnn.trainingError

def correct(inputs, targets):
    correct = []
    for input, target in zip(inputs, targets):
        prediction = rnn.predict(input)
        if prediction.index(max(prediction)) == targets.index(max(targets)):
            correct.append(1)
        else:
            correct.append(0)
    return float(sum(correct))/float(len(correct))

print "training correct:", correct(inputs, targets)


testDataset = 'set1-die-optimal-8.pickle'

inputs, targets, nextProbs = getData(testDataset)

print "test dataset:", testDataset
print "test correct:", correct(inputs, targets)
