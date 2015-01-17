# IOR - Interactive Object Recognition

This repo contains my research in interactive object recognition.

Right now, I'm working on building an unsupervised model to predict the next side
of a die after an action given all previous actions and observations. Thus, we need
a model that is very powerful at learning latent variables, in this case, 3D pose.

## To Do
read about initialization for the ARNN
 - right weights
 - greeduily pretrain!
RNN forward feed layer util.
MLP layer util.
Fresh install on linux machine. Install cuda, etc. Run theano models on there.

correct prediction errors. get hidden units. mlp to predict the object.



pylearn2!
http://deeplearning.net/software/pylearn2/

use ROS for testing implementation?




- supervised train a model to predict the object based on the hidden weights
- iterate through actions, get predictions, predict object, minimum expected entropy, maximum expected entropy
- use a tensor model




- back to my arnn?
- try crossentropy with clipping?


## Getting Started

    pip install sparkprob
    pip install se3
    sudo python dice/setup.py develop


## just some random thoughts

Why dont we think about rnn training as some sort of graph optimization?
Given some inputs, follow the positive weights to the output. 
Define this as a backbone in the RNN that resists change.
Can we use some form of max-flow min-cut?