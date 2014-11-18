#!/usr/bin/python
# coding: utf-8

# (Terrible name right now)
# Autoencoding Recurrent Hidden Markov Model
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


import theano
import theano.tensor as T
import numpy
from sparkprob import sparkprob
import cPickle as pickle

# hide warnings
import warnings
warnings.simplefilter("ignore")

def shuffleInUnison(arr):
    rng_state = numpy.random.get_state()
    for item in arr:
        numpy.random.shuffle(item)
        numpy.random.set_state(rng_state)


class ARNN:
    def __init__(self, n=0, nobs=0, nact=0, L1_reg=0.0, L2_reg=0.0):

        # regularization hyperparameters
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)

        # number of hidden units
        self.n = n
        # number of input units
        self.nobs = nobs
        # number of output units
        self.nact = nact

        print "Constructing the ARNN..."
        self.constructModel()
        print "Constructing the ARNN trainer..."
        self.constructTrainer()

    def constructModel(self):
        """
        Construct the computational graph of the ARNN with Theano.
        """

        # observations (where first dimension is time)
        self.o = T.matrix()
        # actions (where first dimension is time)
        self.a = T.matrix()

        # recurrent (filter) weights as a shared variable
        self.W_hk = theano.shared(numpy.random.uniform(size=(self.n, self.n), low=-.01, high=.01))
        # recurrent (update) weights as a shared variable
        self.W_kh = theano.shared(numpy.random.uniform(size=(self.n, self.n), low=-.01, high=.01))
        # input to hidden layer (encoding) weights
        self.W_ho = theano.shared(numpy.random.uniform(size=(self.n, self.nobs), low=-.01, high=.01))
        # hidden layer to output (dencoding) weights
        self.W_zh = theano.shared(numpy.random.uniform(size=(self.nobs, self.n), low=-.01, high=.01))
        # action to hidden layer weights
        self.W_ka = theano.shared(numpy.random.uniform(size=(self.n, self.nact), low=-.01, high=.01))
        # hidden layer to prediction weights (tied decoding weights)
        self.W_yk = self.W_zh
        # hidden layer bias weights
        self.b_h = theano.shared(numpy.zeros((self.n)))
        # hidden layer bias weights
        self.b_k = theano.shared(numpy.zeros((self.n)))
        # decoder weights
        self.b_z = theano.shared(numpy.zeros((self.nobs)))
        # decoder tied weights
        self.b_y = self.b_z
        # initial hidden state of the ARNN
        self.k0 = theano.shared(numpy.zeros((self.n)))

        self.params = [
            self.W_hk,
            self.W_kh,
            self.W_ho,
            self.W_zh,
            self.W_ka,
            # self.W_yk,
            self.b_h,
            self.b_k,
            self.b_z,
            # self.b_y,
            self.k0
        ]


        # Auto Encoder:
        # p(h_t | k_{t-1}, o_t) = sigmoid(W_hk k_{t-1} + W_ho o_t + b_h)
        # p(z_t | h_t) = sigmoid(W_zh h_t + b_z)
        # J = error(z_t, o_t)
        def autoencodeStep(k_tm1, o_t):
            h_t = T.nnet.sigmoid(T.dot(W_hk, k_tm1) + T.dot(W_ho, o_t) + b_h)
            z_t = T.nnet.sigmoid(T.dot(W_zh, h_t) + b_z)
            return h_t, z_t


        # Predictor:
        # p(k_t | h_t, a_t) = sigmoid(W_kh h_t + W_ka a_t + b_k)
        # p(y_t | k_t) = sigmoid(W_yk k_t + b_y)
        # J = error(y_t, o_{t+1})
        def preditStep(h_t, a_t):
            k_t = T.nnet.sigmoid(T.dot(W_kh, h_t) + T.dot(W_ka, a_t) + b_k)
            y_t = T.nnet.sigmoid(T.dot(W_yk, k_t) + b_y)
            return k_t, y_t



        # https://groups.google.com/forum/#!topic/theano-users/1CWfhf7AqDU











        # recurrent function
        def step(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(self.W_hx, x_t) + T.dot(self.W_hh, h_tm1) + self.b_h)
            y_t = T.nnet.sigmoid(T.dot(self.W_yh, h_t) + self.b_y)
            return h_t, y_t

        # the hidden state `h` for the entire sequence, and the output for the
        # entrie sequence `y` (first dimension is always time)
        [self.h, self.y], _ = theano.scan(step,
                                sequences=self.x,
                                outputs_info=[self.h0, None])

        # predict function outputs y for a given x
        self.predict = theano.function(inputs=[self.x,], outputs=self.y)

    def constructTrainer(self):
        """
        Construct the computational graph of Stochastic Gradient Decent (SGD) on the RNN with Theano.
        """

        # L1 norm
        self.L1 = 0
        self.L1 += abs(self.W_hh.sum())
        self.L1 += abs(self.W_hx.sum())
        self.L1 += abs(self.W_yh.sum())

        # square of L2 norm
        self.L2_sqr = 0
        self.L2_sqr += (self.W_hh ** 2).sum()
        self.L2_sqr += (self.W_hx ** 2).sum()
        self.L2_sqr += (self.W_yh ** 2).sum()

        # error between output and target
        # self.loss = abs(self.y-self.t).sum()
        self.loss = T.mean(T.nnet.binary_crossentropy(self.y, self.t))
        self.cost =  self.loss + self.L1_reg*self.L1  + self.L2_reg*self.L2_sqr

        # gradients on the weights using BPTT
        self.gparams = [T.grad(self.cost, param) for param in self.params]
        self.gradients = theano.function(inputs=[self.x, self.t], outputs=self.gparams)

        # error is a function while cost is the symbolic variable
        self.error = theano.function(inputs=[self.x,self.t], outputs=self.cost)

        # learning rate
        self.lr = T.scalar()

        updates = [
            (param, param - self.lr * gparam)
            for param, gparam in zip(self.params, self.gparams)
        ]

        # training function
        self.trainStep = theano.function([self.x, self.t, self.lr],
                             self.cost,
                             updates=updates)

    def trainModel(self, inputs, targets, learningRate=0.1, epochs=100, shuffle=True):
        """
        Train the RNN on the provided inputs and targets for a given learning rate and number of epochs.
        """

        print "Training the RNN..."
        if shuffle:
            shuffleInUnison([inputs, targets])

        # training
        for epoch in range(epochs):
            print "EPOCH: %d/%d" % (epoch, epochs)
            e = None
            for i in range(len(inputs)):
                e = self.trainStep(inputs[i], targets[i], learningRate)
            print "  ", e
        print "DONE"
        print ""

    def testModel(self, x, t):
        y = self.predict(x)
        print 'x'.center(len(x[0])*2) + '  |  ' + 'y'.center(len(y[0])*2) + '  |  ' + 't'.center(len(y[0])*2)
        print ''
        for i in range(len(y)):
            print sparkprob(x[i]) + '  |  ' + sparkprob(y[i]) + '  |  ' +  sparkprob(t[i])

    def save(self):
        pass

    def load(self):
        pass










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
