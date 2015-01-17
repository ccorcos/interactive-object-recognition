Inspired by deeplearning papers about autoencoders and their success with unsupervised learning on static models, I thought I'd try this method on dynamic systems which will be much more suitable for the robotics of the future that learn based on a time series signal of information. In this work, I explore using recurrent neural networks for unsupervised learning by optimizing the prediction of the next obeservation conditioned on all previous actions and observations. Thus, learning an appropriate internal state and state transition mechanisms for the problem at hand. 

Following in suit with the autoencoder commmunity, we can then use a multilayer perceptron to learn what we want from the internal state. For example, if a robot were to move about a room learning with a LiDaR scanner, it may learn a compact representation of the map it is moving through by being able to accurately predict the next set of LiDaR measurements. Also, the robot would learn its motion model through its own perceptions. Given a sufficient model, the latent internal state that is learned should be able to be trained to easily to give labels for the location of the robot such as which room it is located in. This is much in the same spirit an autoencoder may be used to learn digits and using simple logistic regression on the hidden units or an MLP to get the desired answers out.

Another example would be manipulating an ambigious object -- similar to the hallway problem for SLAM.

The beauty of such a model is that its can be repurposed in several ways. It can be used as a graph search for path planning, exploration, or searching for an object.

It also inherently uses localization. As demonstrated by disambiguating the object. It can do efficient localization by searching for an efficient means of determining its location / what object it is observing.


