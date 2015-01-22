# IOR - Interactive Object Recognition

This repo contains my research in interactive object recognition.

Right now, I'm working on building an unsupervised model to predict the next side
of a die after an action given all previous actions and observations. Thus, we need
a model that is very powerful at learning latent variables, in this case, 3D pose.

## To Do
What if we plug the prediction into the observation and only show the actions.

RNN, ARNN for supervised learning entirely!

RNN forward feed layer util.


pylearn2!
http://deeplearning.net/software/pylearn2/



## Getting Started

    pip install sparkprob
    pip install se3
    sudo python dice/setup.py develop


## just some random thoughts

Why dont we think about rnn training as some sort of graph optimization?
Given some inputs, follow the positive weights to the output. 
Define this as a backbone in the RNN that resists change.
Can we use some form of max-flow min-cut

## Things I've Tried

Use a RNN and ARNN for 