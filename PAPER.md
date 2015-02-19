
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

