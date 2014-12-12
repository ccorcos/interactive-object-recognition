#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy

# hide warnings
import warnings
warnings.simplefilter("ignore")

class Layer:
    def __init__(self, nin=None, nout=None, W=None, b=None, activation=T.tanh, name=''):
        """
        activation:        the activation function for the hidden layer
        name:              name of the hidden layer
        nin, nout, W, b:   either pass the dimensions of the input and output, or pass
                           the precomputed weight and bias symbolic theano tensors.
        """
        self.activation = activation

        if W is None:
            assert(nin is not None)
            assert(nout is not None)
            W = theano.shared(
                numpy.random.uniform(
                    size=(nout, nin),
                    low=-numpy.sqrt(6. / (nin + nout)),
                    high=numpy.sqrt(6. / (nin + nout))
                ),
                name=name+'.W'
            )

        if b is None:
            assert(nin is not None)
            assert(nout is not None)
            b = theano.shared(numpy.zeros((nout,)), name=name+'.b')

        self.W = W
        self.b = b
        self.params = [self.W,self.b]
        self.weights = [W]

    def compute(self, input):
        """
        compute the output for a given input
        """
        linear = T.dot(self.W, input) + self.b

        if self.activation:
            return self.activation(linear)
        else:
            return linear


class MultiInputLayer:
    def __init__(self, nins=[], nout=None, Ws=[], b=None, activation=T.tanh, name=''):
        """
        activation:           the activation function for the hidden layer
        name:                 name of the hidden layer
        nins, nouts, Ws, bs:  either pass the dimensions of the inputs and outputs, or pass
                              the precomputed weight and bias symbolic theano tensors.
        """
        self.activation = activation

        if len(Ws) is 0:
            assert(len(nins) is not 0)
            assert(nout is not None)
            for i in range(len(nins)):
                nin = nins[i]
                W = theano.shared(
                    numpy.random.uniform(
                        size=(nout, nin),
                        low=-numpy.sqrt(6. / (nin + nout)),
                        high=numpy.sqrt(6. / (nin + nout))
                    ),
                    name=name+'['+ str(i) + '].W'
                )
                Ws.append(W)

        if b is None:
            assert(nout is not None)
            b = theano.shared(numpy.zeros((nout,)), name=name+'.b')

        self.Ws = Ws
        self.b = b
        self.params = self.Ws + [b]
        self.weights = Ws

    def compute(self, inputs):
        assert(len(inputs) is len(self.Ws))
        linear = self.b
        for i in range(len(inputs)):
            linear += T.dot(self.Ws[i], inputs[i])

        if self.activation:
            return self.activation(linear)
        else:
            return linear
