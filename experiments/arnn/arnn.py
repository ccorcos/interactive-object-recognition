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


# momentum
# dropout
# stopping condition
# multiple die
import pydot
import theano
import theano.tensor as T
import numpy
from sparkprob import sparkprob
import cPickle as pickle

# hide warnings
import warnings
warnings.simplefilter("ignore")

MODE ='FAST_COMPILE'

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

        self.constructModel()
        self.constructTrainer()

    def constructModel(self):
        """
        Construct the computational graph of the ARNN with Theano.
        """
        print "Constructing the ARNN..."
        # observations (where first dimension is time)
        self.o = T.matrix()
        # actions (where first dimension is time)
        self.a = T.matrix()
        print "  creating weights"
        # recurrent (filter) weights as a shared variable
        self.W_hk = theano.shared(numpy.random.uniform(size=(self.n, self.n), low=-.01, high=.01), name="W_hk")
        # recurrent (update) weights as a shared variable
        self.W_kh = theano.shared(numpy.random.uniform(size=(self.n, self.n), low=-.01, high=.01), name="W_kh")
        # input to hidden layer (encoding) weights
        self.W_ho = theano.shared(numpy.random.uniform(size=(self.n, self.nobs), low=-.01, high=.01), name="W_ho")
        # hidden layer to output (dencoding) weights
        self.W_zh = theano.shared(numpy.random.uniform(size=(self.nobs, self.n), low=-.01, high=.01), name="W_zh")
        # action to hidden layer weights
        self.W_ka = theano.shared(numpy.random.uniform(size=(self.n, self.nact), low=-.01, high=.01), name="W_ka")
        # hidden layer to prediction weights (tied decoding weights)
        # self.W_yk = self.W_zh
        # hidden layer bias weights
        self.b_h = theano.shared(numpy.zeros((self.n)), name="b_h")
        # hidden layer bias weights
        self.b_k = theano.shared(numpy.zeros((self.n)), name="b_k")
        # decoder weights
        self.b_z = theano.shared(numpy.zeros((self.nobs)), name="b_z")
        # decoder tied weights
        # self.b_y = self.b_z
        # initial hidden state of the ARNN
        self.k0 = theano.shared(numpy.zeros((self.n)), name="k0")

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
            h_t = T.nnet.sigmoid(T.dot(self.W_hk, k_tm1) + T.dot(self.W_ho, o_t) + self.b_h)
            z_t = T.nnet.sigmoid(T.dot(self.W_zh, h_t) + self.b_z)
            return h_t, z_t

        # Predictor:
        # p(k_t | h_t, a_t) = sigmoid(W_kh h_t + W_ka a_t + b_k)
        # p(y_t | k_t) = sigmoid(W_yk k_t + b_y)
        # J = error(y_t, o_{t+1})
        def preditStep(h_t, a_t):
            print self.W_kh.shape, h_t.shape
            k_t = T.nnet.sigmoid(T.dot(self.W_kh, h_t) + T.dot(self.W_ka, a_t) + self.b_k)
            y_t = T.nnet.sigmoid(T.dot(self.W_zh, k_t) + self.b_z)
            return k_t, y_t

        # There will always be one more observation than there are actions
        # take care of it upfront so we can use scan
        h0, z0 = autoencodeStep(self.k0, self.o[0])

        def step(a_t, o_t, h_t):
            k_t, y_t = preditStep(h_t, a_t)
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
        T.join(0, h0[numpy.newaxis,:], self.h)
        T.join(0, z0[numpy.newaxis,:], self.z)

        # k = [self.k0]
        # h = []
        # y = []
        # z = []
        # for t in range(self.a.shape[0]:
        #     h_t, z_tp1 = autoencodeStep(k[t], self.o[t])
        #     h.append(h_t)
        #     z.append(z_tp1)
        #     k_t, y_t = preditStep(h[t], self.a[t])
        #     k.append(k_t)
        #     y.append(y_t)
        #
        # t = self.o.shape[0]
        # h_t, z_tp1 = autoencodeStep(k[t], self.o[t])
        # h.append(h_t)
        # z.append(z_tp1)

        # self.k = numpy.array(k)
        # self.h = numpy.array(h)
        # self.y = numpy.array(y)
        # self.z = numpy.array(z)

        print "  compiling the prediction function"
        # predict function outputs y for a given x
        self.predict = theano.function(inputs=[self.o, self.a], outputs=self.y, mode=MODE)
        # theano.printing.pydotprint(self.predict)

    def constructTrainer(self):
        """
        Construct the computational graph of Stochastic Gradient Decent (SGD) on the ARNN with Theano.
        """
        print "Constructing the ARNN trainer..."
        # L1 norm
        self.L1 = 0
        self.L1 += abs(self.W_hk.sum())
        self.L1 += abs(self.W_kh.sum())
        self.L1 += abs(self.W_ho.sum())
        self.L1 += abs(self.W_zh.sum())
        self.L1 += abs(self.W_ka.sum())

        # square of L2 norm
        self.L2_sqr = 0
        self.L2_sqr += (self.W_hk ** 2).sum()
        self.L2_sqr += (self.W_kh ** 2).sum()
        self.L2_sqr += (self.W_ho ** 2).sum()
        self.L2_sqr += (self.W_zh ** 2).sum()
        self.L2_sqr += (self.W_ka ** 2).sum()

        # prediction loss
        self.ploss = T.mean(T.nnet.binary_crossentropy(self.y, self.o[1:]))
        # autoencoding loss
        self.aloss = T.mean(T.nnet.binary_crossentropy(self.z, self.o))

        self.cost =  self.ploss + self.aloss + self.L1_reg*self.L1  + self.L2_reg*self.L2_sqr
        print "  compiling error function"
        # error is a function while cost is the symbolic variable
        self.error = theano.function(inputs=[self.o,self.a], outputs=self.cost, mode=MODE)

        print "  computing the gradient"
        # gradients on the weights using BPTT
        self.gparams = [T.grad(self.cost, param) for param in self.params]
        print "  compiling gradient function"
        self.gradients = theano.function(inputs=[self.o, self.a], outputs=self.gparams, mode=MODE)

        # learning rate
        self.lr = T.scalar()

        updates = [
            (param, param - self.lr * gparam)
            for param, gparam in zip(self.params, self.gparams)
        ]

        print "  compiling training function"
        # training function
        self.trainStep = theano.function([self.o, self.a, self.lr],
                             self.cost,
                             updates=updates,
                             mode=MODE)

    def trainModel(self, observations, actions, learningRate=0.1, epochs=100, shuffle=True):
        """
        Train the ARNN on the provided observations and actions for a given learning rate and number of epochs.
        """
        n_samples = len(observations)

        print observations.shape, actions.shape

        print "Training the ARNN..."
        # if shuffle:
        #     shuffleInUnison([observations, actions])

        # training
        for epoch in range(epochs):
            print "EPOCH: %d/%d" % (epoch, epochs)
            e = None
            for i in range(n_samples):
                e = self.trainStep(observations[i], actions[i], learningRate)
            print "  ", e
        print "DONE"
        print ""

    def testModel(self, obs, act):
        y = self.predict(obs)
        print 'obs'.center(len(obs[0])*2) + '  |  ' + 'y'.center(len(y[0])*2) + '  |  ' + 'act'.center(len(y[0])*2)
        print ''
        for i in range(len(y)):
            print sparkprob(obs[i]) + '  |  ' + sparkprob(y[i]) + '  |  ' +  sparkprob(act[i])

    def save(self):
        pass

    def load(self):
        pass
