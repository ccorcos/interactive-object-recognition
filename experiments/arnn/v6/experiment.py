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


rnn = RNN(
    n_obs=6,
    n_act=5,
    n_hidden=80,
    ff_obs=[50],
    ff_filt=[50],
    ff_trans=[50],
    ff_act=[50],
    ff_pred=[50],
    L1_reg = .01,
    L2_reg = .0,
    transitionActivation=layer.tanh,
    outputActivation=layer.softmax
)

# shuffleInUnison([observations, actions])

# you can only learn one sequence!
rnn.trainModel(
    observations=observations[0:1,:,:],
    actions=actions[0:1,:,:],
    learningRate=0.1,
    momentum=0.1,
    epochs=200
)
rnn.testModel(observations[0], actions[0])


# rnn.predict(observations[0], actions[0])


"""
# What if we do pretraining? The problem looks like an initialization problem.
# So lets train only on the first transition. We can move forward after.


i = observations.shape[0] # number of trials
n = 1 # number of steps
lr = 0.1
mom = 0.2
e = 10 # epochs


steps = observations.shape[1]

# # print "one step of one trial"
# err = rnn.trainModel(observations=observations[0:(i+1),0:(n+1),:],actions=actions[0:(i+1),0:n,:],learningRate=lr,momentum=mom,epochs=e)
# rnn.testModel(observations[i, 0:(n+1), :], actions[i, 0:n, :])

# n = 3
# while n < 20:
#   print "n:", n
#   print "err:", err
#   err = rnn.trainModel(observations=observations[0:(i+1),0:(n),:],actions=actions[0:(i+1),0:(n-1),:],learningRate=lr,momentum=mom,epochs=e)
#   rnn.testModel(observations[i, 0:(n), :], actions[i, 0:(n-1), :])
#   if err < 0.2:
#     n += 1

def randRange(min, max):
  return int(round(random.random()*(max-min)+min))


# Optimize backwards
# Or forwards, it really shouldnt matter

err = rnn.trainModel(observations=observations[0:i,(steps-n-1):,:],actions=actions[0:i,(steps-n-1):,:],learningRate=lr,momentum=mom,epochs=e)
rnn.testModel(observations[i-1, (steps-n-1):, :], actions[i-1, (steps-n-1):, :])

n = 2
while n < steps:
  if err > 1:
    lr *= 1.1
    lr = min(lr, 0.9)
    mom *= 0.9
    mom = max(mom, 0.1)
  else: 
    lr *= 0.95
    lr = max(lr, 0.1)
    mom *= 1.1
    mom = min(mom, 0.9)
  print "n:", n
  print "err:", err
  err = rnn.trainModel(observations=observations[0:i,(steps-n-1):,:],actions=actions[0:i,(steps-n-1):,:],learningRate=lr,momentum=mom,epochs=e)
  rnn.testModel(observations[randRange(0, i-1), (steps-n-1):, :], actions[randRange(0, i-1),(steps-n-1):, :])
  if err < 0.4:
    n += 1

"""