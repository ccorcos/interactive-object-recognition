#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy

# hide warnings
import warnings
warnings.simplefilter("ignore")

class Layer:
    def __init__(self, input, nin=None, nout=None, W=None, b=None, activation=T.tanh, name=''):
        """
        input:             a theano symbolic vector
        activation:        the activation function for the hidden layer
        name:              name of the hidden layer
        nin, nout, W, b:   either pass the dimensions of the input and output, or pass
                           the precomputed weight and bias symbolic theano tensors.
        """
        self.input = input
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

        linear = T.dot(self.W, self.input) + self.b

        if self.activation:
            self.output = self.activation(linear)
        else:
            self.output = linear


class MultiInputLayer:
    def __init__(self, inputs=[], nins=[], nout=None, Ws=[], b=None, activation=T.tanh, name=''):
        """
        inputa:               an array of theano symbolic vectors
        activation:           the activation function for the hidden layer
        name:                 name of the hidden layer
        nins, nouts, Ws, bs:  either pass the dimensions of the inputs and outputs, or pass
                              the precomputed weight and bias symbolic theano tensors.
        """
        n = len(inputs)
        assert(n is not 0)
        self.inputs = inputs
        self.activation = activation

        if len(Ws) is 0:
            assert(len(nins) is n)
            assert(nout is not None)
            for i in range(n):
                input = inputs[i]
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

        linear = self.b
        for i in range(n):
            linear += T.dot(self.Ws[i], self.inputs[i])

        if self.activation:
            self.output = self.activation(linear)
        else:
            self.output = linear
