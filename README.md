# IOR - Interactive Object Recognition

This repo contains my research in interactive object recognition.

Right now, I'm working on building an unsupervised model to predict the next side
of a die after an action given all previous actions and observations. Thus, we need
a model that is very powerful at learning latent variables, in this case, 3D pose.

## To Do

- try rnn with softmax?
- hmm?
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
