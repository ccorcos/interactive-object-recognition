# IOR - Interactive Object Recognition

This repo contains my research in interactive object recognition.

Right now, I'm working on building an unsupervised model to predict the next side
of a die after an action given all previous actions and observations. Thus, we need
a model that is very powerful at learning latent variables, in this case, 3D pose.

## To Do

HMM stuff:
traditionally, hmm's just have a single number at the output which is the
probability of the next. If they all sum to 1, then we have a finite state
automata. This would make sense for me if i had poses\*objects number of hidden
states. However, we could instead use a softmax.

When do you use the viterbi algorithm here? For tracing back which previous
states you we actually in...?

Anyways, you could use simple sigmoid instead of softmax -- this would allow for
a combinatorial state vector.



Backpropagation through time seems like a totally valid way to train these
anyways.

1. simple hmm with unit values that sum to zero
2. simple hmm with softmax
3. simple hmm with sigmoid units


- hmm
- my model
- dropout?
- minibatches
- momentum
- stopping criteria

- get the rnn to learn just one die
- get the rnn to learn a set of dice
- write up BACKGROUND.md and RNN.md
- everything again on my own model
- everything again with a HMM

## Getting Started

    pip install sparkprob
    pip install se3
    sudo python dice/setup.py develop
