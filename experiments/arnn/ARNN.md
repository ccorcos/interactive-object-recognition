I have no good name for this alrigthm but I'm alling it an autoencoding recurrent neural network.

It is very similar to a neural network but has an interest Kalman-filter like motivation.

The network is split into two different parts. The autoencoding part approximates a multimodal distribution
of the hidden state. The action transforms that multimodal distribution, and then the next observation
filters that distribution again.



I think this model and the RNN would greatly benefit from momentum and dropout.


# V1

simple implementation. doesnt really work

# V2

refactored. deep implementation. still has its problems


Naming:

This is a lot like a kalman filter for robotics. Its also a recurrent neural network because it repeats and exhibits the markov property. It also uses autoencoding in a very interesting way.

Autoencoding Kalman Filter Recurrent Neural Network

Its also deep with deep input, output and hidden transitions.




Theano Performance.

How can I test whether Theano is properly leveraging my GPU and multiple CPU cores?
http://deeplearning.net/software/theano/tutorial/multi_cores.html
http://deeplearning.net/software/theano/tutorial/using_gpu.html

Is T.grad supposed to leverage this?

How can I save a model / compiled functions so I dont have to wait forever to run it again, etc?
http://deeplearning.net/software/theano/tutorial/loading_and_saving.html

How do I run large models? Use sklearn tools? Run on a remote cluster?

pylearn2!
http://deeplearning.net/software/pylearn2/