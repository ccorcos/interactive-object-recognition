# IOR - Interactive Object Recognition

This repo contains my research in interactive object recognition.

Right now, I'm working on building an unsupervised model to predict the next side
of a die after an action given all previous actions and observations.

## Getting Started

    pip install sparkprob
    pip install se3
    cd dice
    python setup.py develop

This package also depends on my [Deep Learning repo](https://github.com/ccorcos/deep-learning).
After cloning it, make sure you setup and develop that package as well. 

You can clean up with

    python setup.py develop --uninstall

## To Do


Generate a dataset. Use the DL package tools and formatting

- n_examples x n_timesteps x n_dim
- observations, actions


Learn one dice using embedding
Learn one dice from multiple poses using embedding

Learn two dice using embedding
Learn a second dice by only training on the embedding matrix