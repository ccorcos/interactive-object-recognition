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


# V4

Tried some incremental training. And experimented with the cost function a little bit. No cigar.

        # the problem here is that we must fo up to come down. so we need to reshape this 
        # surface so that doing a little worse could mean doing a little better.
        # guessing all zeros leads to an error of 1/5 on the output which is a ploss of 1.
        # guessing a value everytime at random is worst case going go give you and error of 2/5
        # on for each step. So we want to incentivise guessing at least once and hopefully we'll
        # find a nice solution from there.
        
        # compensate = -0.5 + abs(1 - self.y.sum(axis=1).mean())
        # compensate = -1 + abs(1 - self.y.sum(axis=1).mean())
        # compensate = T.switch(T.gt(compensate,0), 0, compensate)

        compensate = 0

# v5

relu works, but sometimes nan more often
softmax definitely helps
tried incremental optimization

# v6

What if we only give actions and optimize the observation predictions?
This would only work for one dice. but suppose we condition on the first n observations and expect a correct prediction for the rest?