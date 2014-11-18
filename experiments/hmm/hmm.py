#!/usr/bin/python
# coding: utf-8


# Hidden Markov Modell
# https://gist.github.com/jzelner/4267380






# x_t: observation at time t
# y_t: predicted observation at time t
# h_t: hidden layer at time t
# a_t: is the action at time t
#
# http://vimdoc.sourceforge.net/htmldoc/digraph.html#digraph-table
# <C-k>Dt: ▼
# <C-k>UT: ▲
# <C-k>PR: ▶
# <C-k>PL: ◀
#
#             a_t             a_{t+1}
#              |               |
#      (W_ha)  |               |
#              |               |
#              ▼    (W_hh)     ▼
#  ---------▶ h_t ---------▶ h_{t+1} ---------▶
#              |               |
#      (W_xh)  |               |
#              |               |
#              ▼               ▼
#             x_t             x_{t+1}
#
#
# Predictor:
# p(h_t | h_{t-1}, a_t) = sigmoid(W_hh h_{t-1} + W_ha a_t + b_h)
# p(x_t | h_t) = sigmoid(W_xh h_t + b_x)
# J = error(y_t, x_{t_1})


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


class HMM:
    def __init__(self, n=0, nx=0, na=0, L1_reg=0.0, L2_reg=0.0):
        """
        Initialize a Hidden Markov Model with state, observation and actions
        sizes and regularization hyperparameters.
        """
        # regularization hyperparameters
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)

        # number of hidden units
        self.n = n
        # number of input units
        self.nx = nx
        # number of output units
        self.na = na

        print "Constructing the HMM..."
        self.constructModel()
        print "Constructing the HMM trainer..."
        self.constructTrainer()

    def constructModel(self):
        """
        Construct the computational graph of the HMM with Theano.
        """

        # observatoins (where first dimension is time)
        self.x = T.matrix()
        # actions (where first dimension is time)
        self.a = T.matrix()

        # recurrent weights as a shared variable
        self.W_hh = theano.shared(numpy.random.uniform(size=(self.n, self.n), low=-.01, high=.01))
        # hidden state to observation weights
        self.W_xh = theano.shared(numpy.random.uniform(size=(self.nx, self.n), low=-.01, high=.01))
        # action to hidden state weights
        self.W_ha = theano.shared(numpy.random.uniform(size=(self.n, self.na), low=-.01, high=.01))
        # hidden layer bias weights
        self.b_h = theano.shared(numpy.zeros((self.n)))
        # observation bias weights
        self.b_x = theano.shared(numpy.zeros((self.nout)))
        # initial hidden state
        self.h0 = theano.shared(numpy.zeros((self.n)))

        self.params = [self.W_hh, self.W_xh, self.W_ha, self.b_h, self.b_x, self.h0]

        def emissionStep(h_t):
            y_t = T.nnet.sigmoid( T.dot(self.W_xh, h_t) + self.b_x)
            return y_t

        def transitionStep(h_tm1, a_t):
            h_t = T.nnet.sigmoid( T.dot(self.W_hh, h_tm1) + T.dot(self.W_ha, a_t) + self.b_h)
            return h_t



        y0 = T.nnet.sigmoid( T.dot(W_xh, h0) + b_x)

        # training like this isnt exactly going to work. We NEED to use EM algorithm for this.
        # maybe an advantage of my model? its all forward-feed.




# HERE








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
