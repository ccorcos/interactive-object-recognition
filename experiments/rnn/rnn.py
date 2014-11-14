#!/usr/bin/python
# coding: utf-8

# Recurrent Neural Network
# This is an implementation of a RNN for unsupervised learning of time series data.
#
# x_t: input at time t
# h_t: hidden layer at time t
# y_t: is the prediction
# t_t: is the targets that y needs to be
#
# http://vimdoc.sourceforge.net/htmldoc/digraph.html#digraph-table
# <C-k>Dt: ▼
# <C-k>UT: ▲
# <C-k>PR: ▶
# <C-k>PL: ◀
#
#
#        (W_hx)          (W_yh)
#  x_t ---------▶  h_t ---------▶ y_t
#
#                   ▲
#           (W_hh)  |
#                   |
#  h_{t-1} ---------
#
#
# Predictor:
# p(h_t | h_{t-1}, x_t) = \sigma(W_hh h_{t-1} + W_hx x_t + b_h)
# p(y_t | h_t) = \sigma(W_yh h_t + b_y)
# J = error(y_t, x_{t_1})
#

import theano
import theano.tensor as T
import numpy
from sparkprob import sparkprob
import cPickle as pickle


# stochastic gradient decent
# use momentum?
# minibatch training
# randomize the sample training order
# generalize parameter updating and gradients -- see comment at the bottom

class RNN:
    def __init__(self, n=0, nin=0, nout=0, L1_reg=0.0, L2_reg=0.0):
        """
        Initialize a Recurrent Nerual Network layer sizes and regularization hyperparameters.

        Example Udage:

        rnn = RNN(
            n = 50,
            nin = 11,
            nout = 6,
            L1_reg = 0,
            L2_reg = 0
        )

        rnn.trainModel(
            inputs=inputs,
            target=target,
            learningRage=0.1,
            L1_reg=0.1,
            L2_reg=0.2
        )

        rnn.testModel(inputs[0], targets[0])

        """
        # regularization hyperparameters
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)

        # number of hidden units
        self.n = n
        # number of input units
        self.nin = nin
        # number of output units
        self.nout = nout

        print "Constructing the RNN..."
        self.constructModel()
        print "Constructing the RNN trainer..."
        self.constructTrainer()

    def constructModel(self):
        """
        Construct the computational graph of the RNN with Theano.
        """

        # input (where first dimension is time)
        self.x = T.matrix()
        # target (where first dimension is time)
        self.t = T.matrix()

        # recurrent weights as a shared variable
        self.W_hh = theano.shared(numpy.random.uniform(size=(self.n, self.n), low=-.01, high=.01))
        # input to hidden layer weights
        self.W_hx = theano.shared(numpy.random.uniform(size=(self.n, self.nin), low=-.01, high=.01))
        # hidden to output layer weights
        self.W_yh = theano.shared(numpy.random.uniform(size=(self.nout, self.n), low=-.01, high=.01))
        # hidden layer bias weights
        self.b_h = theano.shared(numpy.zeros((self.n)))
        # output layer bias weights
        self.b_y = theano.shared(numpy.zeros((self.nout)))
        # initial hidden state of the RNN
        self.h0 = theano.shared(numpy.zeros((self.n)))

        # recurrent function
        def step(x_t, h_tm1, W_hh, W_hx, W_yh, b_h, b_y):
            h_t = T.nnet.sigmoid(T.dot(W_hx, x_t) + T.dot(W_hh, h_tm1) + b_h)
            y_t = T.nnet.sigmoid(T.dot(W_yh, h_t) + b_y)
            return h_t, y_t

        # the hidden state `h` for the entire sequence, and the output for the
        # entrie sequence `y` (first dimension is always time)
        [self.h, self.y], _ = theano.scan(step,
                                sequences=self.x,
                                outputs_info=[self.h0, None],
                                non_sequences=[self.W_hh, self.W_hx, self.W_yh, self.b_h, self.b_y])

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
        self.loss = abs(self.y-self.t).sum()
        self.err =  self.loss + self.L1_reg*self.L1  + self.L2_reg*self.L2_sqr

        # gradients on the weights using BPTT
        self.gW_hh, self.gW_hx, self.gW_yh, self.gb_h, self.gb_y, self.gh0 = T.grad(self.err, [self.W_hh, self.W_hx, self.W_yh, self.b_h, self.b_y, self.h0])
        self.gradients = theano.function(inputs=[self.x, self.t], outputs=[self.gW_hh, self.gW_hx, self.gW_yh, self.gb_h, self.gb_y, self.gh0])

        # error is a function while err is the symbolic variable
        self.error = theano.function(inputs=[self.x,self.t], outputs=self.err)

        # learning rate
        self.lr = T.scalar()

        # training function
        self.trainStep = theano.function([self.x, self.t, self.lr],
                             self.err,
                             updates={self.W_hh: self.W_hh - self.lr * self.gW_hh,
                                      self.W_hx: self.W_hx - self.lr * self.gW_hx,
                                      self.W_yh: self.W_yh - self.lr * self.gW_yh,
                                      self.b_h: self.b_h - self.lr * self.gb_h,
                                      self.b_y: self.b_y - self.lr * self.gb_y,
                                      self.h0: self.h0 - self.lr * self.gh0})

    def trainModel(self, inputs, targets, learningRate=0.1, epochs=100):
        """
        Train the RNN on the provided inputs and targets for a given learning rate and number of epochs.
        """

        print "Training the RNN..."
        # training
        for epoch in range(epochs):
            print "EPOCH: %d/%d" % (epoch, epochs)
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


# # W_kh, b_y are ommited because they're redundant
# params = [W_ka, W_ho, W_zh, W_kh, W_hk, b_h, b_z, b_k, k0]
#
# # L1
# L1 = 0
# for param in params:
# 	L1 += abs(param.sum())
#
# # square of L2 norm
# L2_sqr = 0
# for param in params:
# 	L2_sqr += (param ** 2).sum()
#
# error = predictError + autoencodeError + L1_reg*L1  + L2_reg*L2_sqr
#
# # gradients on the weights using BPTT
# gParams = T.grad(error, params)
#
# gradients = theano.function(inputs=[o,a], outputs=gParams)
# err = theano.function(inputs=[o,a], outputs=error)
#
# updates = {}
# for param, gParam in zip(params, gParams):
# 	updates[param] = param - lr * gParam
#
# # training function
# trainStep = theano.function([o, a, lr],
#                      error,
#                      updates=updates)
