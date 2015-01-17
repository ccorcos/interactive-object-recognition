#!/usr/bin/python
# coding: utf-8

# Recurrent Neural Network
# This is an implementation of a RNN for unsupervised learning of time series data.
# This network has sigmoid transfer functions and binary outputs. And it uses


import theano
import theano.tensor as T
import numpy
from sparkprob import sparkprob
import cPickle as pickle
from layer import *
import operator

# hide warnings
import warnings
warnings.simplefilter("ignore")


def shuffleInUnison(arr):
    rng_state = numpy.random.get_state()
    for item in arr:
        numpy.random.shuffle(item)
        numpy.random.set_state(rng_state)


class RNN:
    def __init__(self, n=0, m=0, nin=0, min=0, nout=0, mout=0, L1_reg=0.0, L2_reg=0.0, transitionActivation=T.nnet.sigmoid, outputActivation=T.nnet.sigmoid):
        """
        Initialize a Recurrent Nerual Network layer sizes and regularization hyperparameters.

        m is the size of the hidden layers at the input, output and transition

        Example Udage:

        rnn = RNN(
            n = 50,
            m = 60,
            nin = 11,
            min = 30,
            nout = 6,
            mout = 20,
            L1_reg = 0,
            L2_reg = 0
        )

        rnn.trainModel(
            inputs=inputs,
            targets=targets,
            learningRate=0.1,
            momentum=0.2,
            epochs=100
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
        # number of hidden transition units
        self.m = m
        # number of hidden input units
        self.min = min
        # number of hidden output units
        self.mout = mout

        self.transitionActivation = transitionActivation
        self.outputActivation = outputActivation

        print "Constructing the RNN..."
        self.constructModel()
        print "Constructing the RNN trainer..."
        self.constructTrainer()

    def constructModel(self):
        """
        Construct the computational graph of the RNN with Theano.

        This we will embed a hidden layer at the input, output, and transition.


                       (W_hx)          (W_yh)
         x_t ---------▶  h_t ---------▶ y_t

                          ▲
                  (W_hh)  |
                          |
         h_{t-1} ---------


         Layer sizes:

         Deep:      Input      Transition              Output
         (nin) ----▶ (min) ----▶  (m) ----▶  (n) ----▶ (mout) ----▶ (nout)

                                  ▲
                                  |
                                  |
                  (n_{t-1}) ------       

        """

        self.inputHiddenLayer = Layer(  nin=self.nin,
                                        nout=self.min,
                                        activation=self.transitionActivation,
                                        name='inputHiddenLayer')


        # the input and the hidden layer go into the input layer
        self.transitionHiddenLayer = MultiInputLayer(   nins=[self.min, self.n],
                                                        nout=self.m,
                                                        activation=self.transitionActivation,
                                                        name='transitionHiddenLayer')

        self.hiddenLayer = Layer(   nin=self.m,
                                    nout=self.n,
                                    activation=self.transitionActivation,
                                    name='hiddenLayer')

        self.outputHiddenLayer = Layer( nin=self.n,
                                        nout=self.mout,
                                        activation=self.transitionActivation,
                                        name='outputHiddenLayer')

        # the hidden layer output goes to the output
        self.outputLayer = Layer( nin=self.mout,
                             nout=self.nout,
                             activation=self.outputActivation,
                             name='outputLayer')

        self.layers = [self.inputHiddenLayer, self.hiddenLayer, self.transitionHiddenLayer, self.outputHiddenLayer, self.outputLayer]

        def step(x_t, h_tm1):
            xp_t = self.inputHiddenLayer.compute(x_t)
            z_t = self.transitionHiddenLayer.compute([xp_t,h_tm1])
            h_t = self.hiddenLayer.compute(z_t)
            yp_t = self.outputHiddenLayer.compute(h_t)
            y_t = self.outputLayer.compute(yp_t)
            return h_t, y_t

        # next we need to scan over all steps for a given array of observations
        # input (where first dimension is time)
        self.x = T.matrix()
        self.x.tag.test_value = numpy.random.rand(50, 11)
        # initial hidden state of the RNN
        self.h0 = theano.shared(numpy.zeros((self.n)))

        # the hidden state `Hs` for the entire sequence, and the output for the
        # entrie sequence `Ys` (first dimension is always time)
        # scan passes the sequenct as the first input and the output info as the rest of the inputs
        [self.h, self.y], _ = theano.scan(step,
                                sequences=self.x,
                                outputs_info=[self.h0, None])

        # gather all the parameters
        self.params = reduce(operator.add, map(lambda x: x.params, self.layers)) + [self.h0]
        # gather all weights, the parameters without the bias terms
        self.weights = reduce(operator.add, map(lambda x: x.weights, self.layers))
        # for every parameter, we maintain it's last update to user gradient decent with momentum
        self.momentums = [theano.shared(numpy.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX)) for param in self.params]

        # predict function outputs y for a given x
        self.predict = theano.function(inputs=[self.x,], outputs=self.y)

    def constructTrainer(self):
        """
        Construct the computational graph of Stochastic Gradient Decent (SGD) on the RNN with Theano.
        """

        # L1 norm
        self.L1 = reduce(operator.add, map(lambda x: abs(x.sum()), self.weights))

        # square of L2 norm
        self.L2_sqr = reduce(operator.add, map(lambda x: (x ** 2).sum(), self.weights))

        # target (where first dimension is time)
        self.t = T.matrix()
        self.t.tag.test_value = numpy.random.rand(50, 6)

        # error between output and target
        self.loss = T.mean(T.nnet.binary_crossentropy(self.y, self.t))
        # self.loss = T.mean(abs(self.y - self.t))
        self.cost =  self.loss + self.L1_reg*self.L1  + self.L2_reg*self.L2_sqr

        # gradients on the weights using BPTT
        self.gparams = [T.grad(self.cost, param) for param in self.params]
        # self.gradients = theano.function(inputs=[self.x, self.t], outputs=self.gparams)

        self.costf = theano.function(inputs=[self.x,self.t], outputs=self.cost)

        # def correctPrediction(yi, ti):
        #     if T.argmax(yi) == T.argmax(ti):
        #         return 1.
        #     else:
        #         return 0.

        # [self.correct,], _ = theano.scan(correctPrediction,
        #                         sequences=[self.y, self.t])

        # self.correct = [correctPrediction(yi,ti) for (yi,ti) in zip(self.y, self.t)]

        # self.correctf = theano.function(inputs=[self.y,self.t], outputs=self.correct)


        # learning rate
        self.lr = T.scalar()
        # momentum
        self.mom = T.scalar()

        updates = []
        for param, gparam, momentum in zip(self.params, self.gparams, self.momentums):
            update = self.mom * momentum - self.lr * gparam
            updates.append((momentum, update))
            updates.append((param, param + update))

        # training function
        self.trainStep = theano.function([self.x, self.t, self.lr, self.mom],
                             self.cost,
                             updates=updates)

    def trainModel(self, inputs, targets, learningRate=0.1, momentum=0.2, epochs=100, shuffle=True):
        """
        Train the RNN on the provided inputs and targets for a given learning rate and number of epochs.
        """

        print "Training the RNN..."
        if shuffle:
            shuffleInUnison([inputs, targets])

        self.trainingError = None
        # training
        for epoch in range(epochs):
            print "EPOCH: %d/%d" % (epoch, epochs)
            for i in range(len(inputs)):
                self.trainingError = self.trainStep(inputs[i], targets[i], learningRate, momentum)
            print "  ", self.trainingError
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
