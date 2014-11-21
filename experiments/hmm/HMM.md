
# Hidden Markov Model (HMM)

Traditionally, HMM's just have a single number at the output which is the probability of the state. The sum of likelihoods of the next state must add up to 1 to ensure that the states are mutually exclusive and add up to 1 as well. This  means we have a finite state automata. This would make for the dice problem if there were poses \* objects number of hidden states. This would quickly become intractable, as we add many object and many poses to be learned. Also, I imagine handling continuous pose by simply expanding the hidden state and treating everything the same. Thus, it is necessary that the hidden state encode object and pose in a factorial way.

One idea here is to use softmax between the hidden states. This would ensure everything adds up to 1. But the state space would still be exponential. Another approach could be to use sigmoid transfer functions between hidden states. This could definitely be factorial but would also require special treatment in terms of training.

I'm curious if it is possible to use backprogagation through time to train a HMM. I've seen articles about it but none of them get terribly specific about it. This would be very convenient for Theano. But I'm not sure its possible. The problem is that an observation will never change the hidden state. Thus we would be optimizing blindly.

But we'll have to take a look at the math.

$$
p(o_t|h_t) = \sigma(W_{oh} h_t + b_o)
p(h_{t+1}|h_t) = \sigma(W_{hh} h_t + b_h)
$$

Now I suppose we could do backprogation through time, but the challenging part is that we need to compute the gradient at each step conditioned on all previous observations. This ought to be pretty challenging in Theano.

Final thoughts: I'm not sure its possible to formulate a hidden markov model as a neural network with softmax
or sigmoid transfer functions because you can't compute the probability of an observation based on a single
hidden unit.
