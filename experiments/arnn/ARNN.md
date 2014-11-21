I have no good name for this alrigthm but I'm alling it an autoencoding recurrent neural network.

It is very similar to a neural network but has an interest Kalman-filter like motivation.

The network is split into two different parts. The autoencoding part approximates a multimodal distribution
of the hidden state. The action transforms that multimodal distribution, and then the next observation
filters that distribution again.



I think this model and the RNN would greatly benefit from momentum and dropout.
