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
Running Experiment 4:
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

"""

                              o_t              a_t
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

relu = lambda x: x * (x > 0)
cappedrelu =  lambda x: T.minimum(x * (x > 0), 6)


rnn = RNN(
    n_obs=6,
    n_act=5,
    n_hidden=30,
    ff_obs=[20],
    ff_filt=[20],
    ff_trans=[30],
    ff_act=[20],
    ff_pred=[20],
    L1_reg = .0,
    L2_reg = .01,
    transitionActivation=relu,
    outputActivation=T.nnet.sigmoid
)

# shuffleInUnison([observations, actions])

# def train(observations, actions):
#   rnn.trainModel(
#       observations=observations,
#       actions=actions,
#       learningRate=0.1,
#       momentum=0.8,
#       epochs=10
#   )



# What if we do pretraining? The problem looks like an initialization problem.
# So lets train only on the first transition. We can move forward after.


# train on just one transition: this is hard because its so inconsistent.
# train(observations[:,0:2,:], actions[:,0:1,:])

# train on just one sequence.
# train(observations[0:1,:,:], actions[0:1,:,:])

# rnn.predict(observations[0], actions[0])

i = 0 # number of trials
n = 1 # number of steps
lr = 0.1
mom = 0.2
e = 20 # epochs

print "one step of one trial"
rnn.trainModel(observations=observations[0:(i+1),0:(n+1),:],actions=actions[0:(i+1),0:n,:],learningRate=lr,momentum=mom,epochs=e)
rnn.testModel(observations[i, 0:(n+1), :], actions[i, 0:n, :])

print "two steps of one trial"
n = 2
rnn.trainModel(observations=observations[0:(i+1),0:(n+1),:],actions=actions[0:(i+1),0:n,:],learningRate=lr,momentum=mom,epochs=e)
rnn.testModel(observations[i, 0:(n+1), :], actions[i, 0:n, :])

for n in range(3, 10):
  print rnn.weights[0].get_value()
  rnn.trainModel(observations=observations[0:(i+1),0:(n+1),:],actions=actions[0:(i+1),0:n,:],learningRate=lr,momentum=mom,epochs=e)
  rnn.testModel(observations[i, 0:(n+1), :], actions[i, 0:n, :])
