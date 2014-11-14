#!/usr/bin/python
# coding: utf-8

# Recurrent Neural Network
# This is an implementation of a RNN for unsupervised learning of an
# action-observation model.
#
# o_t: observation at time t
# h_t: hidden layer at time t
# y_t: is the prediction
#
# http://vimdoc.sourceforge.net/htmldoc/digraph.html#digraph-table
# <C-k>Dt: ▼
# <C-k>UT: ▲
# <C-k>PR: ▶
# <C-k>PL: ◀
#
#
#  o_t -------------
#                   |
#           (W_ho)  |
#                   ▼
#        (W_ha)          (W_yh)
#  a_t ---------▶  h_t ---------▶ y_t
#
#                   ▲
#           (W_hh)  |
#                   |
#  h_{t-1} ---------
#
#
# Predictor:
# p(h_t | h_{t-1}, o_t, a_t) = \sigma(W_hh h_{t-1} + W_ho o_t + W_ha a_t + b_h)
# p(y_t | h_t) = \sigma(W_yh h_t + b_y)
# J = error(y_t, o_{t_1})
#

import theano
import theano.tensor as T
import numpy
import cPickle as pickle
from sparkprob import sparkprob


with open('data.pickle', 'rb') as handle:
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

# Using action[t] + observation[t] as input, try to predict observations[t+1]
# nextProbs will give you the actual probability of the obervations[t+1]

inputs = []
outputs = []
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
    outputs.append(o)
    nextProbs.append(p)

inputs = numpy.array(inputs, dtype=theano.config.floatX)
outputs = numpy.array(outputs, dtype=theano.config.floatX)
nextProbs = numpy.array(nextProbs, dtype=theano.config.floatX)











L1_reg = 0.0
L2_reg = 0.0

# number of hidden units
n = 50
# number of input units
nin = 11
# number of output units
nout = 6

# input (where first dimension is time)
x = T.matrix()
# target (where first dimension is time)
t = T.matrix()
# learning rate
lr = T.scalar()

# recurrent weights as a shared variable
W_hh = theano.shared(numpy.random.uniform(size=(n, n), low=-.01, high=.01))
# input to hidden layer weights
W_hx = theano.shared(numpy.random.uniform(size=(n, nin), low=-.01, high=.01))
# hidden to output layer weights
W_yh = theano.shared(numpy.random.uniform(size=(nout, n), low=-.01, high=.01))
# hidden layer bias weights
b_h = theano.shared(numpy.zeros((n)))
# output layer bias weights
b_y = theano.shared(numpy.zeros((nout)))
# initial hidden state of the RNN
h0 = theano.shared(numpy.zeros((n)))

# recurrent function
def step(x_t, h_tm1, W_hh, W_hx, W_yh, b_h, b_y):
    h_t = T.nnet.sigmoid(T.dot(W_hx, x_t) + T.dot(W_hh, h_tm1) + b_h)
    y_t = T.nnet.sigmoid(T.dot(W_yh, h_t) + b_y)
    return h_t, y_t


# the hidden state `h` for the entire sequence, and the output for the
# entrie sequence `y` (first dimension is always time)
[h, y], _ = theano.scan(step,
                        sequences=x,
                        outputs_info=[h0, None],
                        non_sequences=[W_hh, W_hx, W_yh, b_h, b_y])

# predict function outputs y for a given x
predict = theano.function(inputs=[x,], outputs=y)

# L1 norm
L1 = 0
L1 += abs(W_hh.sum())
L1 += abs(W_hx.sum())
L1 += abs(W_yh.sum())

# square of L2 norm
L2_sqr = 0
L2_sqr += (W_hh ** 2).sum()
L2_sqr += (W_hx ** 2).sum()
L2_sqr += (W_yh ** 2).sum()

# error between output and target
# error = T.mean(T.nnet.binary_crossentropy(y, t)) + L1_reg*L1  + L2_reg*L2_sqr
error = abs(y-t).sum() + L1_reg*L1  + L2_reg*L2_sqr

# gradients on the weights using BPTT
gW_hh, gW_hx, gW_yh, gb_h, gb_y, gh0 = T.grad(error, [W_hh, W_hx, W_yh, b_h, b_y, h0])

gradients = theano.function(inputs=[x,t], outputs=[gW_hh, gW_hx, gW_yh, gb_h, gb_y, gh0])
err = theano.function(inputs=[x,t], outputs=error)

# training function
trainStep = theano.function([x, t, lr],
                     error,
                     updates={W_hh: W_hh - lr * gW_hh,
                              W_hx: W_hx - lr * gW_hx,
                              W_yh: W_yh - lr * gW_yh,
                              b_h: b_h - lr * gb_h,
                              b_y: b_y - lr * gb_y,
                              h0: h0 - lr * gh0})









for j in range(20):
    for i in range(len(inputs)):
        print trainStep(inputs[i],outputs[i],0.000001)

for index in range(15):
    print "\n\n"
    predicted = predict(inputs[index])
    for i in range(len(predicted)):
        print sparkprob.sparkprob(predicted[i]) + '   ' +  sparkprob.sparkprob(nextProbs[index][i])



for j in range(200):
    print trainStep(inputs[0],outputs[0], 0.01)



predicted = predict(inputs[0])
for i in range(len(predicted)):
    print sparkprob.sparkprob(prediction) + '   ' +  sparkprob.sparkprob(nextProbs[index][i])


for j in range(20):
    for i in range(len(inputs)):
        print trainStep(inputs[i],nextProbs[i],0.00001)


predict(inputs[index])
index = 1
err(inputs[index], outputs[index])
gradients(inputs[index], outputs[index])
trainStep(inputs[index], outputs[index], 0.01)

T.nnet.binary_crossentropy(predict(inputs[index]), outputs[index])
