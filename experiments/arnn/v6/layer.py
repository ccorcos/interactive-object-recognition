#!/usr/bin/python
# coding: utf-8

import theano
import theano.tensor as T
import numpy
import operator

# hide warnings
import warnings
warnings.simplefilter("ignore")

relu = lambda x: T.switch(x<0, 0, x)
cappedrelu =  lambda x: T.minimum(T.switch(x<0, 0, x), 6)
sigmoid = T.nnet.sigmoid
tanh = T.tanh

def softmax(x):
    e_x = T.exp(x - x.max())
    return e_x / e_x.sum()


class Layer:
    def __init__(self, nin=None, nout=None, activation=tanh, name=''):
        """
        activation:        the activation function for the hidden layer
        name:              name of the hidden layer
        nin, nout          the dimensions of the input and output
        """
        self.activation = activation

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
    def __init__(self, nins=[], nout=None, activation=tanh, name=''):
        """
        activation:           the activation function for the hidden layer
        name:                 name of the hidden layer
        nins, nouts:          the dimensions of the inputs and outputs
        """
        self.activation = activation

        assert(len(nins) is not 0)
        assert(nout is not None)
        Ws = []
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



class ForwardFeed:
    def __init__(self, n=[], activation=tanh, outputActivation=None, name=''):
        """
        activation:        the activation function for the hidden layer
        name:              name of the hidden layer
        n:                 the dimensions of the layers
        """

        if outputActivation is None:
            self.outputActivation = activation
        else:
            self.outputActivation = outputActivation

        self.activation = activation
        self.n = n
        self.layers = []
        self.params = []
        self.weights = []

        assert(len(n) is not 0)
        if len(n) > 1:
            for i in range(0, len(n)-2):
                nin = n[i]
                nout = n[i+1]
                layer = Layer( nin=nin,
                               nout=nout,
                               activation=self.activation,
                               name=name + '.layers[' + str(i) + ']')
                self.layers.append(layer)
            
            # output
            i = len(n)-2
            nin = n[i]
            nout = n[i+1]
            layer = Layer( nin=nin,
                           nout=nout,
                           activation=self.outputActivation,
                           name=name + '.layers[' + str(i) + ']')
            self.layers.append(layer)
            
            self.params = reduce(operator.add, map(lambda x: x.params, self.layers))
            self.weights = reduce(operator.add, map(lambda x: x.weights, self.layers))

    def compute(self, input):
        """
        compute the output for a given input
        """

        output = input
        for layer in self.layers:
            output = layer.compute(output)
        return output


        