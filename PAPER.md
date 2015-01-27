Inspired by deeplearning papers about autoencoders and their success with unsupervised learning on static models, I thought I'd try this method on dynamic systems which will be much more suitable for the robotics of the future that learn based on a time series signal of information. In this work, I explore using recurrent neural networks for unsupervised learning by optimizing the prediction of the next obeservation conditioned on all previous actions and observations. Thus, learning an appropriate internal state and state transition mechanisms for the problem at hand. 

Following in suit with the autoencoder commmunity, we can then use a multilayer perceptron to learn what we want from the internal state. For example, if a robot were to move about a room learning with a LiDaR scanner, it may learn a compact representation of the map it is moving through by being able to accurately predict the next set of LiDaR measurements. Also, the robot would learn its motion model through its own perceptions. Given a sufficient model, the latent internal state that is learned should be able to be trained to easily to give labels for the location of the robot such as which room it is located in. This is much in the same spirit an autoencoder may be used to learn digits and using simple logistic regression on the hidden units or an MLP to get the desired answers out.

Another example would be manipulating an ambigious object -- similar to the hallway problem for SLAM.

The beauty of such a model is that its can be repurposed in several ways. It can be used as a graph search for path planning, exploration, or searching for an object.

It also inherently uses localization. As demonstrated by disambiguating the object. It can do efficient localization by searching for an efficient means of determining its location / what object it is observing.




------------------------------------------------------------

"Unsupervised State Estimation using Deep Learning for Interactive Object Recognition"

Outline:

Deep learning:
 - unsupervised learning
 - autoencoders for pretraining
 - recurrent neural networks
 - recurrent autoencoder for ASR

Goal:
 - unsupervised learning for state estimation
 - pretraining for supervising learning on hidden state

Models:
 - RNN for unsupervised
 - ARNN for unsupervised

train to predict the next observation. using the hidden state varaibles, supervised predict the die from h_t. For each action, given what we're expected to see next, compute the likelihood for each die. Take the action that leads to the minimum entropy over these guesses. This is the optimal.

Likely overfit. User regularizatoin and dropout.

 - RNN for supervised
 - ARNN for supervised

Predict the object likelihood directly as opposed to this unsupervised middleman. performance?

Other Problems:

Try to use this model on LiDaR SLAM to predict the room and navigate between rooms.

