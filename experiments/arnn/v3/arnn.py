#!/usr/bin/python
# coding: utf-8

# Autoencoding Kalman Filter Recurrent Neural Network
#

# TODO:
# momentum training
# stopping condition
# graphic visualization
# implementation to actually be used
# better metrics for evaluating
# dropout
# try on other problems like SLAM

import theano
import theano.tensor as T
import numpy
from sparkprob import sparkprob
import operator
from layer import *

# hide warnings
import warnings
warnings.simplefilter("ignore")

MODE ='FAST_COMPILE'

class RNN:
    def __init__(self, n_enc=[], n_dec=[], n_act=[], n_hid=[], L1_reg=0.0, L2_reg=0.0):
        """
        INPUTS:
        n_enc:  layer sizes for the encoder
        n_dec:  layer sizes for the decoder
        n_act:  layer sizes for the actions
        n_hid:  layer sizes for the hidden transitions

        MODEL:
        o_t: observation at time t
        z_t: is the observation reconstruction
        y_t: is the prediction
        h_t: the state representation
        k_t: the predictive state representation


                              o_t              a_t
                               |                |
                               |                |
                       encoder |         action |
                               |                |
                               |                |
                    filter     ▼   transform    ▼
         k_{t-1} ----------▶  h_t ----------▶  k_t ----------▶  h_{t+1}
                               |                |
                               |                |
                       decoder |      predictor |
                               |                |
                               |                |
                               ▼                ▼
                              z_t              y_t


        LOSS:
        Auto Encoder:   loss = distance(z_t, o_t)
        Predictor:      loss = distance(y_t, o_{t+1})

        """
        # regularization hyperparameters
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)

        self.n_enc = n_enc
        self.n_dec = n_dec
        self.n_act = n_act
        self.n_hid = n_hid

        assert(n_hid[0] is n_dec[0])

        self.constructModel()
        self.constructTrainer()

    def constructModel(self):
        """
        Construct the computational graph of the RNN with Theano.
        """
        
        print "Constructing the RNN..."

        # intialize the forward feed layers
        self.encoderFF   = ForwardFeed(n=self.n_enc, activation=T.tanh, name='encoderFF')
        self.decoderFF   = ForwardFeed(n=self.n_dec, activation=T.tanh, name='decoderFF')
        self.predictorFF = ForwardFeed(n=self.n_dec, activation=T.tanh, name='predictorFF')
        self.transformFF = ForwardFeed(n=self.n_hid, activation=T.tanh, name='transformFF')
        self.actionFF    = ForwardFeed(n=self.n_act, activation=T.tanh, name='actionFF')
        self.filterFF    = ForwardFeed(n=self.n_hid, activation=T.tanh, name='filterFF')

        # initialize the h and k layers to connect all the forward feed layers
        self.hiddenStateLayer =  MultiInputLayer(nins=[self.n_hid[-1], self.n_enc[-1]],
                                                 nout=self.n_hid[0],
                                                 activation=T.tanh,
                                                 name='hiddenStateLayer')
        
        self.predictiveStateLayer =  MultiInputLayer(nins=[self.n_hid[-1], self.n_act[-1]],
                                                     nout=self.n_hid[0],
                                                     activation=T.tanh,
                                                     name='predictiveStateLayer')

        self.layers = [self.encoderFF, self.decoderFF, self.predictorFF, self.actionFF, self.filterFF, self.hiddenStateLayer, self.predictiveStateLayer]

        def autoencodeStep(k_tm1, o_t):
            h_t = self.hiddenStateLayer.compute([self.filterFF.compute(k_tm1), self.encoderFF.compute(o_t)])
            z_t = self.decoderFF.compute(h_t)
            return h_t, z_t


        def predictStep(h_t, a_t):
            k_t = self.predictiveStateLayer.compute([self.transformFF.compute(h_t), self.actionFF.compute(a_t)])
            y_t = self.predictorFF.compute(k_t)
            return k_t, y_t


        # observations (where first dimension is time)
        self.o = T.matrix()
        self.o.tag.test_value = numpy.random.rand(51, 6)
        # actions (where first dimension is time)
        self.a = T.matrix()
        self.a.tag.test_value = numpy.random.rand(50, 5)

        # initial predictive state of the RNN
        self.k0 = theano.shared(numpy.zeros((self.n_hid[0])), name="k0")

        # There will always be one more observation than there are actions
        # take care of it upfront so we can use scan
        h0, z0 = autoencodeStep(self.k0, self.o[0])

        def step(a_t, o_t, h_t):
            k_t, y_t = predictStep(h_t, a_t)
            h_tp1, z_tp1 = autoencodeStep(k_t, o_t)
            return h_tp1, k_t, y_t, z_tp1

        print "  creating graph"
        [self.h, self.k, self.y, self.z], _ = theano.scan(step,
            sequences=[self.a, self.o[1:]],
            outputs_info=[h0, None, None, None])

        # self.h should be (t by n) dimension.
        # h0 should be of dimension (n)
        # therefore h0[numpy.newaxis,:] should have shape (1 by n)
        # we can use join to join them to create a (t+1 by n) matrix

        # tack on the lingering h0 and z0
        self.h = T.join(0, h0[numpy.newaxis,:], self.h)
        self.z = T.join(0, z0[numpy.newaxis,:], self.z)

        print "  compiling the prediction function"
        # predict function outputs y for a given x
        self.predict = theano.function(inputs=[self.o, self.a], outputs=self.y, mode=MODE)

    def constructTrainer(self):
        """
        Construct the computational graph of Stochastic Gradient Decent (SGD) on the RNN with Theano.
        """

        print "Constructing the RNN trainer..."

        # gather all the parameters
        self.params = reduce(operator.add, map(lambda x: x.params, self.layers)) + [self.k0]
        # gather all weights, the parameters without the bias terms
        self.weights = reduce(operator.add, map(lambda x: x.weights, self.layers))
        # for every parameter, we maintain it's last update to user gradient decent with momentum
        self.momentums = [theano.shared(numpy.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX)) for param in self.params]

        # Use mean instead of sum to be more invariant to the network size and the minibatch size.
        
        # L1 norm
        self.L1 = reduce(operator.add, map(lambda x: abs(x.mean()), self.weights))
        # square of L2 norm
        self.L2_sqr = reduce(operator.add, map(lambda x: (x ** 2).mean(), self.weights))

        # prediction loss
        self.ploss = abs(self.y-self.o[1:]).mean()
        # self.ploss = T.mean(T.nnet.binary_crossentropy(self.y, self.o[1:]))
        # autoencoding loss
        self.aloss = abs(self.z-self.o).mean()
        # self.aloss = T.mean(T.nnet.binary_crossentropy(self.z, self.o))

        self.cost =  self.ploss + self.aloss + self.L1_reg*self.L1  + self.L2_reg*self.L2_sqr
        
        # print "  compiling error function"
        # error is a function while cost is the symbolic variable
        # self.error = theano.function(inputs=[self.o,self.a], outputs=self.cost, mode=MODE)

        print "  computing the gradient"
        # gradients on the weights using BPTT
        self.gparams = [T.grad(self.cost, param) for param in self.params]
        
        # print "  compiling gradient function"
        # self.gradients = theano.function(inputs=[self.o, self.a], outputs=self.gparams, mode=MODE)

        # learning rate
        self.lr = T.scalar()
        self.lr.tag.test_value = 0.1
        # momentum
        self.mom = T.scalar()
        self.mom.tag.test_value = 0.2

        updates = []
        for param, gparam, momentum in zip(self.params, self.gparams, self.momentums):
            update = self.mom * momentum - self.lr * gparam
            updates.append((momentum, update))
            updates.append((param, param + update))

        print "  compiling training function"
        # training function
        self.trainStep = theano.function([self.o, self.a, self.lr, self.mom],
                             self.cost,
                             updates=updates,
                             mode=MODE)

    def trainModel(self, observations, actions, learningRate=0.1, momentum=0.2, epochs=100):
        """
        Train the RNN on the provided observations and actions for a given learning rate and number of epochs.
        """
        n_samples = len(observations)

        print "Training the RNN..."

        # training
        for epoch in range(epochs):
            print "EPOCH: %d/%d" % (epoch, epochs)
            e = None
            for i in range(n_samples):
                e = self.trainStep(observations[i], actions[i], learningRate, momentum)
            print "  ", e
        print "DONE"
        print ""

    def testModel(self, obs, act):
        y = self.predict(obs, act)
        print 'obs'.center(len(obs[0])*2) + '  |  ' + 'y'.center(len(y[0])*2) + '  |  ' + 'act'.center(len(y[0])*2)
        print ''
        for i in range(len(y)):
            print sparkprob(obs[i]) + '  |  ' + sparkprob(y[i]) + '  |  ' +  sparkprob(act[i])

    def save(self):
        pass

    def load(self):
        pass
