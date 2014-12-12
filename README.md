# IOR - Interactive Object Recognition

This repo contains my research in interactive object recognition.

Right now, I'm working on building an unsupervised model to predict the next side
of a die after an action given all previous actions and observations. Thus, we need
a model that is very powerful at learning latent variables, in this case, 3D pose.

## To Do
- refactor v6 with mlp
- try with relu's, tanh's, different initialization, softmax
- back to my arnn



- perhaps try a softmax with a simple rnn
- try a deep rnn

- test the scanop on the compiled theano function in v3
- get v4 working with all the classes
- v5 add deeper layers!


- dropout is for overfitting... not for underfitting
- try relu's for faster training?
- try deeper layers.



## Getting Started

    pip install sparkprob
    pip install se3
    sudo python dice/setup.py develop
