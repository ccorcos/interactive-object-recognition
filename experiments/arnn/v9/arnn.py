#!/usr/bin/python
# coding: utf-8

# Kalman Filter-like Recurrent Neural Network

import theano
import theano.tensor as T
import numpy
from sparkprob import sparkprob
import operator
from layer import *
import cPickle as pickle
from stopwatch import *

# hide warnings
import warnings
warnings.simplefilter("ignore")

# MODE ='FAST_COMPILE'
MODE ='FAST_RUN'
# MODE='DebugMode'

fmt = lambda x: "{:12.8f}".format(x)


class RNN:
    def __init__(self):
        self.time = Stopwatch(name="RNN")
    
    @classmethod
    def create(cls, 
               warmUp=5,
               L1_reg=0.0, 
               L2_reg=0.0, 
               n_obs=0,
               n_act=0,
               n_hidden=0,
               ff_obs=[],
               ff_filt=[],
               ff_trans=[],
               ff_act=[],
               ff_pred=[],
               transitionActivation='sigmoid', 
               outputActivation='sigmoid'):
        """

        INPUTS:
        n_obs: number of observation inputs
        n_act: number of action inputs
        n_hidden: number of hidden state nodes
        ff_obs: an array of layer sizes for the deep input
        ff_filt: an array of layer sizes for the deep transition
        ff_trans: an array of layer sizes for the deep transition
        ff_act: an array of layer sizes for the deep input
        ff_pred: an array of layer sizes for the deep output
       
        MODEL:
        o_t: observation at time t
        a_t: action at time t
        y_t: is the prediction
        h_t: the state representation
        k_t: the predictive state representation


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
        rnn = cls()
        # regularization hyperparameters
        rnn.L1_reg = float(L1_reg)
        rnn.L2_reg = float(L2_reg)

        rnn.warmUp = warmUp
        rnn.n_obs = n_obs
        rnn.n_act = n_act
        rnn.n_hidden = n_hidden
        rnn.ff_obs = ff_obs
        rnn.ff_filt = ff_filt
        rnn.ff_trans = ff_trans
        rnn.ff_act = ff_act
        rnn.ff_pred = ff_pred

        rnn.transitionActivation = transitionActivation
        rnn.outputActivation = outputActivation

        rnn.constructModel()
        rnn.constructTrainer()
        return rnn

    @classmethod
    def load(cls, filename):

        print "Loading from saved file: " + filename
        f = open(filename, 'rb')
        data = pickle.load(f)
        f.close()
        print "  loaded"

        rnn = cls()

        print ""
        print "model parameters:"
        for key in ['n_hidden', 'ff_obs', 'ff_act', 'ff_pred', 'ff_filt', 'ff_trans', 'transitionActivation', 'outputActivation']:
            print key + ": " + str(data[key])
        print ""
        print "training parameters:"
        for key in ['L1_reg','L2_reg', 'lr', 'mom', 'epochs', 'warmUp', 'dataset']:
            print key + ": " + str(data[key])
        # probably print out what was loaded...
        print ""
        print data['notes']
        print ""

        rnn.warmUp = data['warmUp']
        rnn.L1_reg = data['L1_reg']
        rnn.L2_reg = data['L2_reg']
        rnn.n_obs = data['n_obs']
        rnn.n_act = data['n_act']
        rnn.n_hidden = data['n_hidden']
        rnn.ff_obs = data['ff_obs']
        rnn.ff_filt = data['ff_filt']
        rnn.ff_trans = data['ff_trans']
        rnn.ff_act = data['ff_act']
        rnn.ff_pred = data['ff_pred']

        rnn.transitionActivation = data['transitionActivation']
        rnn.outputActivation = data['outputActivation']

        rnn.saved = data['saved']

        rnn.k0_init = data['k0']
        rnn.momentums_init = data['momentums']

        rnn.constructModel()
        rnn.constructTrainer()

        return rnn, data['lr'], data['mom'], data['dataset'], data['epochs'], data['notes']

    def save(self, filename, lr=None, mom=None, dataset=None, epochs=None, notes=None):
        # pass the learning rate, momentum, and dataset used to train it for future reference.
        print "Saving model to file: " + filename
        saved = {
            'warmUp': self.warmUp,
            'L1_reg': self.L1_reg,
            'L2_reg': self.L2_reg,
            'n_obs': self.n_obs,
            'n_act': self.n_act,
            'n_hidden': self.n_hidden,
            'ff_obs': self.ff_obs,
            'ff_filt': self.ff_filt,
            'ff_trans': self.ff_trans,
            'ff_act': self.ff_act,
            'ff_pred': self.ff_pred,
            'transitionActivation': self.transitionActivation,
            'outputActivation': self.outputActivation,
            'saved': {
                'observationFF': self.observationFF.save(),
                'filterFF': self.filterFF.save(),
                'actionFF': self.actionFF.save(),
                'transformFF': self.transformFF.save(),
                'predictorFF': self.predictorFF.save(),
                'hiddenStateLayer': self.hiddenStateLayer.save(),
                'predictiveStateLayer': self.predictiveStateLayer.save(),
            },
            'k0': self.k0.get_value().tolist(),
            'momentums': map(lambda m: m.get_value().tolist(), self.momentums),
            'lr': lr,
            'mom': mom,
            'dataset': dataset, 
            'notes': notes,
            'epochs': epochs
        }

        f = open(filename, 'wb')
        pickle.dump(saved, f,  protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print "  saved"


    def constructModel(self):
        """
        Construct the computational graph of the RNN with Theano.
        """
        
        print "Constructing the RNN..."
        print self.time
        print "  initializing layers"

        # intialize the forward feed layers
        self.observationFF = ForwardFeed(n=[self.n_obs]+self.ff_obs, 
                                         activation=self.transitionActivation,
                                         name='observationFF',
                                         saved=maybe(lambda: self.saved['observationFF']))

        self.filterFF = ForwardFeed(n=[self.n_hidden]+self.ff_filt, 
                                    activation=self.transitionActivation, 
                                    name='filterFF',
                                    saved=maybe(lambda: self.saved['filterFF']))

        self.actionFF = ForwardFeed(n=[self.n_act]+self.ff_act, 
                                    activation=self.transitionActivation, 
                                    name='actionFF',
                                    saved=maybe(lambda: self.saved['actionFF']))

        self.transformFF = ForwardFeed(n=[self.n_hidden]+self.ff_trans, 
                                       activation=self.transitionActivation, 
                                       name='transformFF',
                                       saved=maybe(lambda: self.saved['transformFF']))
        
        self.predictorFF = ForwardFeed(n=[self.n_hidden]+self.ff_pred+[self.n_obs], 
                                       activation=self.transitionActivation, 
                                       outputActivation=self.outputActivation,
                                       name='predictorFF',
                                       saved=maybe(lambda: self.saved['predictorFF']))


        # initialize the h and k layers to connect all the forward feed layers

        self.hiddenStateLayer =  MultiInputLayer(nins=[([self.n_hidden]+self.ff_filt)[-1], ([self.n_obs]+self.ff_obs)[-1]],
                                                 nout=self.n_hidden,
                                                 activation=self.transitionActivation,
                                                 name='hiddenStateLayer',
                                                 saved=maybe(lambda: self.saved['hiddenStateLayer']))
        
        self.predictiveStateLayer =  MultiInputLayer(nins=[([self.n_hidden]+self.ff_trans)[-1], ([self.n_act]+self.ff_act)[-1]],
                                                     nout=self.n_hidden,
                                                     activation=self.transitionActivation,
                                                     name='predictiveStateLayer',
                                                     saved=maybe(lambda: self.saved['predictiveStateLayer']))

        self.layers = [self.observationFF, self.transformFF, self.predictorFF, self.actionFF, self.filterFF, self.hiddenStateLayer, self.predictiveStateLayer]

        def observeStep(k_tm1, o_t):
            h_t = self.hiddenStateLayer.compute([self.filterFF.compute(k_tm1), self.observationFF.compute(o_t)])
            return h_t


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

        # view the first warmUp observations, then be able to halucinate the rest!

        # initial k0, the prior
        if maybe(lambda: self.k0_init):
            self.k0_init = numpy.array(self.k0_init)
            assert(self.k0_init.shape[0] is self.n_hidden)
            self.k0 = theano.shared(self.k0_init, name="k0")
        else:
            self.k0 = theano.shared(numpy.zeros((self.n_hidden)), name="k0")

        def step(a_t, o_tm1, k_tm1):
            h_t = observeStep(k_tm1, o_tm1)
            k_t, y_t = predictStep(h_t, a_t)
            return y_t, k_t, h_t

        print "  creating computational graph"
        print self.time

        # step through the first i observations
        [y1, k1, h1], _ = theano.scan(step,
            sequences=[self.a[0:self.warmUp, :], self.o[0:self.warmUp, :]],
            outputs_info=[None, self.k0, None])

        # with the most recent k and y, halicunate the rest
        [y2, k2, h2], _ = theano.scan(step,
            sequences=[self.a[self.warmUp:, :]],
            outputs_info=[y1[-1], k1[-1], None])

        # self.h should be (t by n) dimension.
        # h0 should be of dimension (n)
        # therefore h0[numpy.newaxis,:] should have shape (1 by n)
        # we can use join to join them to create a (t+1 by n) matrix

        # attach the segments together
        self.k = T.join(0, self.k0[numpy.newaxis,:], k1)
        self.k = T.join(0, self.k, k2)

        self.h = T.join(0, h1, h2)

        self.y = T.join(0, y1, y2)

        self.ypred = T.argmax(self.y, axis=1)
        self.opred = T.argmax(self.o[1:], axis=1)

        # self.error = T.mean(T.neq(self.ypred, self.opred))
        self.error = T.mean(T.neq(self.ypred[self.warmUp:], self.opred[self.warmUp:]))

        print "  compiling the prediction function"
        print self.time
        # predict function outputs y for a given x
        self.predict = theano.function(inputs=[self.o, self.a], outputs=[self.y, self.ypred, self.error], mode=MODE)

    def constructTrainer(self):
        """
        Construct the computational graph of Stochastic Gradient Decent (SGD) on the RNN with Theano.
        """

        print "Constructing the RNN trainer..."
        print self.time

        # gather all the parameters
        self.params = reduce(operator.add, map(lambda x: x.params, self.layers)) + [self.k0]
        # gather all weights, the parameters without the bias terms
        self.weights = reduce(operator.add, map(lambda x: x.weights, self.layers))
        # for every parameter, we maintain it's last update to user gradient decent with momentum

        if maybe(lambda: self.momentums_init):
            self.momentums = [theano.shared(numpy.array(m)) for m in self.momentums_init]
        else:
            self.momentums = [theano.shared(numpy.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX)) for param in self.params]

        # Use mean instead of sum to be more invariant to the network size and the minibatch size.
        
        # L1 norm
        self.L1 = reduce(operator.add, map(lambda x: abs(x).mean(), self.weights))
        # square of L2 norm
        self.L2_sqr = reduce(operator.add, map(lambda x: (x ** 2).mean(), self.weights))

        if self.outputActivation is 'softmax':
            self.ploss = T.mean(T.nnet.binary_crossentropy(self.y[self.warmUp:],self.o[self.warmUp+1:]))
        else:
            # prediction loss, normalized to 1. 0 is optimal. 1 is naive. >1 is just wrong.
            self.ploss = T.mean(abs(self.y[self.warmUp:]-self.o[self.warmUp+1:]))*self.n_obs

        self.cost =  self.ploss + self.L1_reg*self.L1  + self.L2_reg*self.L2_sqr
        

        print "  computing the gradient"
        print self.time
        # gradients on the weights using BPTT
        self.gparams = [T.grad(self.cost, param) for param in self.params]
        
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
        print self.time
        printY = theano.printing.Print("y:")
        printO = theano.printing.Print("o[1:]:")
        printLoss = theano.printing.Print("ploss:")
        # training function
        self.trainStep = theano.function([self.o, self.a, self.lr, self.mom],
                             [self.cost, self.error],
                             updates=updates,
                             mode=MODE)

    def train(self, observations, actions, learningRate=0.1, momentum=0.2, epochs=100):
        """
        Train the RNN on the provided observations and actions for a given learning rate and number of epochs.
        """
        n_samples = len(observations)

        print "Training the RNN..."
        print self.time
        cost = 100
        error = 100
        # training
        for epoch in range(epochs):
            print "EPOCH: %d/%d" % (epoch+1, epochs)
            for i in range(n_samples):
                cost, error = self.trainStep(observations[i], actions[i], learningRate, momentum)
            print "  cost: " + fmt(float(cost)) + " error: " + fmt(float(error))
        print "DONE"
        print self.time
        print ""
        return cost, error

    def visualize(self, obs, act):
        print "Visualizing trial..."
        y, ypred, error = self.predict(obs, act)
        print 'obs'.center(len(obs[0])*2) + '  |  ' + 'y'.center(len(y[0])*2) + '  |  ' + 'act'.center(len(y[0])*2)
        print ''
        print sparkprob(obs[0]) + '  |  ' + sparkprob([0]*6) + '  |  ' +  sparkprob(act[0])
        for i in range(1, len(act)):
            print sparkprob(obs[i]) + '  |  ' + sparkprob(y[i-1]) + '  |  ' +  sparkprob(act[i])
        print sparkprob(obs[len(obs)-1]) + '  |  ' + sparkprob(y[len(obs)-2]) + '  |  ' + sparkprob([0]*5)
        print "error: " + str(error)

    def test(self, observations, actions):
        print "Testing on a dataset..."
        results = numpy.array(map(lambda obs, act: self.predict(obs, act), observations, actions))
        y = results[:,0]
        ypred = results[:,1]
        error = results[:,2]
        e = error.mean()
        print "error: " + str(e)
        return e

