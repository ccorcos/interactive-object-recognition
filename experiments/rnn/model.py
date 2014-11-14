#!/usr/bin/python
# coding: utf-8

# Conditional Deep Recurrent Denoising Auto Encoder
#
# Inspired by the DRDAE for ASR:
# Recurrent Neural Networks for Noise Reduction in Robust ASR 2012
# http://www1.icsi.berkeley.edu/~vinyals/Files/rnn_denoise_2012.pdf
#
# o_t: observation at time t
# h_t: hidden layer in the autoencoder
# z_t: is the reconstruction
# k_t: hidden layer in the action-observation predictor
# y_t: is the prediction
#
# http://vimdoc.sourceforge.net/htmldoc/digraph.html#digraph-table
# <C-k>Dt: ▼
# <C-k>UT: ▲
# <C-k>PR: ▶
# <C-k>PL: ◀
#
#                       o_t              a_t
#                        |                |
#                        |                |
#                 (W_ho) |         (W_ka) |
#                        |                |
#                        |                |
#            (W_hk)      ▼    (W_kh)      ▼
#  k_{t-1} ----------▶  h_t ----------▶  k_t ----------▶  h_{t+1}
#                        |                |
#                        |                |
#                 (W_zh) |         (W_yk) |
#                        |                |
#                        |                |
#                        ▼                ▼
#                       z_t              y_t
#
#
# Auto Encoder:
# p(h_t | k_{t-1}, o_t) = \sigma(W_hk k_{t-1} + W_ho o_t + b_h)
# p(z_t | h_t) = \sigma(W_zh h_t + b_z)
# J = error(z_t, o_t)
#
# Predictor:
# p(k_t | h_t, a_t) = \sigma(W_kh h_t + W_ka a_t + b_k)
# p(y_t | k_t) = \sigma(W_yk k_t + b_y)
# J = error(y_t, o_{t+1})
#
# Tied weights:
# W_yk = W_zh and b_y = b_z enforces h and k to be a similar representation.
# W_ho = W_zh makes for less computation. Any substatial reason for this?
# W_kh =? W_hk could be interesting. The k only differs from h with a_t vs o_t.
#
# Weight Names:
# W_action: W_ka
# W_encode: W_ho
# W_decode: W_zh, W_yk
# W_transfrom: W_kh
# W_filter: W_hk
#

# Notes / Thoughts:
# How can we make sure that we don't get a zero issue like the RNN? Perhaps we
# need to have a lower bound on the regularizations limit?



import theano
import theano.tensor as T
import numpy
from utils import *

dot = numpy.dot
sigmoid = T.nnet.sigmoid
crossEntropy = T.nnet.binary_crossentropy

# rng = numpy.random.RandomState(1234)

# Model Parameters
L1_reg = 0.1
L2_reg = 0.1
n_observations = 10
n_actions = 3
n_hidden = 20

# learning rate
lr = T.scalar()

W_ka = theano.shared(value=randomWeights(n_hidden, n_actions), name='W_ka')
W_ho = theano.shared(value=randomWeights(n_hidden, n_observations), name='W_ho')
# W_zh = W_ho.T
W_zh = theano.shared(value=randomWeights(n_observations, n_hidden), name='W_zh')
W_yk = W_zh.T
W_kh = theano.shared(value=randomWeights(n_hidden, n_hidden), name='W_kh')
W_hk = theano.shared(value=randomWeights(n_hidden, n_hidden), name='W_hk')

b_h = theano.shared(value=zeroVector(n_hidden), name='b_h')
b_z = theano.shared(value=zeroVector(n_hidden), name='b_z')
b_y = b_z
b_k = theano.shared(value=zeroVector(n_hidden), name='b_k')

k0 = theano.shared(value=zeroVector(n_hidden), name='k0')

# Model Structure
# There is always one more observation than action. To account for this,
# there must be an extra action at the end, but it doesnt matter what this is.
o = T.matrix('o') # observations (time x n_observations)
a = T.matrix('a') # actions (time x n_actions)

# Predictor:
# p(k_t | h_t, a_t) = \sigma(W_kh h_t + W_ka a_t + b_k)
# p(y_t | k_t) = \sigma(W_yk k_t + b_y)
# J = error(y_t, o_{t+1})
def preditStep(h_t, a_t, W_kh, W_ka, W_yk, b_k, b_y):
	k_t = sigmoid(dot(W_kh, h_t) + dot(W_ka, a_t) + b_k)
	y_t = sigmoid(dot(W_yk, k_t) + b_y)
	return k_t, y_t

# Auto Encoder:
# p(h_t | k_{t-1}, o_t) = \sigma(W_hk k_{t-1} + W_ho o_t + b_h)
# p(z_t | h_t) = \sigma(W_zh h_t + b_z)
# J = error(z_t, o_t)
def autoencodeStep(k_tm1, o_t, W_hk, W_ho, W_zh, b_h, b_z):
	for p in [k_tm1, o_t, W_hk, W_ho, W_zh, b_h, b_z]:
		print p.type

	h_t = sigmoid(dot(W_hk, k_tm1) + dot(W_ho, o_t) + b_h)
	z_t = sigmoid(dot(W_zh, h_t) + b_z)
	return h_t, z_t

# Step:
# There will be one more observation than action. Thus the last action doesnt
# matter, but don't forget it!
def step(o_t, a_t, k_tm1, W_hk, W_ho, W_zh,  W_kh, W_ka, W_yk, b_k, b_y, b_h, b_z):
	h_t, z_t = autoencodeStep(k_tm1, o_t, W_hk, W_ho, W_zh, b_h, b_z)
	k_t, y_t = preditStep(h_t, a_t, W_kh, W_ka, W_yk, b_k, b_y)
	return h_t, z_t, k_t, y_t


# Scan over each step for the observations and actions.
[h, z, k, y], _ = theano.scan(step,
						sequences=[o, a],
						outputs_info=[None, None, k0, None],
						non_sequences=[W_hk, W_ho, W_zh, W_kh, W_ka, W_yk, b_k, b_y, b_h, b_z])

print 'here'









# predict function outputs y for given o and a
predict = theano.function(inputs=[o,a], outputs=y)

# auctoencode function outputs z for given o and a
autoencode = theano.function(inputs=[o,a], outputs=z)

# predictError = crossEntropy(y[:-1], o[1:])
predictError = abs(y[:-1] - o[1:]).sum()

# autoencodeError = crossEntropy(z,o)
autoencodeError = abs(z-o).sum()

# W_kh, b_y are ommited because they're redundant
params = [W_ka, W_ho, W_zh, W_kh, W_hk, b_h, b_z, b_k, k0]

# L1
L1 = 0
for param in params:
	L1 += abs(param.sum())

# square of L2 norm
L2_sqr = 0
for param in params:
	L2_sqr += (param ** 2).sum()

error = predictError + autoencodeError + L1_reg*L1  + L2_reg*L2_sqr

# gradients on the weights using BPTT
gParams = T.grad(error, params)

gradients = theano.function(inputs=[o,a], outputs=gParams)
err = theano.function(inputs=[o,a], outputs=error)

updates = {}
for param, gParam in zip(params, gParams):
	updates[param] = param - lr * gParam

# training function
trainStep = theano.function([o, a, lr],
                     error,
                     updates=updates)

#
#
# for j in range(20):
#     for i in range(len(inputs)):
#         print trainStep(inputs[i],outputs[i],0.000001)
#
#
# for index in range(15):
#     print nn
#     predicted = predict(inputs[index])
#     for i in range(len(predicted)):
#         print sparklines.sparklines(predicted[i]) + '   ' +  sparklines.sparklines(nextProbs[index][i])
#
#
#
# for j in range(200):
#     print trainStep(inputs[0],outputs[0], 0.01)
#
#
#
# predicted = predict(inputs[0])
# for i in range(len(predicted)):
#     print sparklines.sparklines(prediction) + '   ' +  sparklines.sparklines(nextProbs[index][i])
#
#
# for j in range(20):
#     for i in range(len(inputs)):
#         print trainStep(inputs[i],nextProbs[i],0.00001)
#
#
# predict(inputs[index])
# index = 1
# err(inputs[index], outputs[index])
# gradients(inputs[index], outputs[index])
# trainStep(inputs[index], outputs[index], 0.01)
#
# T.nnet.binary_crossentropy(predict(inputs[index]), outputs[index])
