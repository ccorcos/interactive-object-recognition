import theano
import theano.tensor as T
import numpy


# The model basically looks like this:
#
#                       o_t              a_t
#                        |                |
#                        |                |
#                 (W_ho) |         (W_ka) |
#                        |                |
#                        |                |
#            (W_hk)      v    (W_kh)      v
#  k_{t-1} ---------->  h_t ---------->  k_t ---------->  h_{t+1}
#                        |                |
#                        |                |
#                 (W_zh) |         (W_yk) |
#                        |                |
#                        |                |
#                        v                v
#                       z_t              y_t


n = 24      # number of elements in h and k
nobs = 6    # number of elements in o, z and y
nact = 5    # number of elements in a

# Symbolic variables for o and a
o = T.matrix()
a = T.matrix()

# The weight matrices
W_hk = theano.shared(numpy.random.uniform(size=(n, n), low=-.01, high=.01), name="W_hk")
W_kh = theano.shared(numpy.random.uniform(size=(n, n), low=-.01, high=.01), name="W_kh")
W_ho = theano.shared(numpy.random.uniform(size=(n, nobs), low=-.01, high=.01), name="W_ho")
W_zh = theano.shared(numpy.random.uniform(size=(nobs, n), low=-.01, high=.01), name="W_zh")
W_ka = theano.shared(numpy.random.uniform(size=(n, nact), low=-.01, high=.01), name="W_ka")
W_yk = W_zh

# The bias vectors
b_h = theano.shared(numpy.zeros((n)), name="b_h")
b_k = theano.shared(numpy.zeros((n)), name="b_k")
b_z = theano.shared(numpy.zeros((nobs)), name="b_z")
b_y = b_z

# the prior
k0 = theano.shared(numpy.zeros((n)), name="k0")


def observationStep(k_tm1, o_t):
    h_t = T.nnet.sigmoid(T.dot(W_hk, k_tm1) + T.dot(W_ho, o_t) + b_h)
    z_t = T.nnet.sigmoid(T.dot(W_zh, h_t) + b_z)
    return h_t, z_t

def actionPredictionStep(h_t, a_t):
    k_t = T.nnet.sigmoid(T.dot(W_kh, h_t) + T.dot(W_ka, a_t) + b_k)
    y_t = T.nnet.sigmoid(T.dot(W_yk, k_t) + b_y)
    return k_t, y_t

def step(h_t, a_t, o_t):
    k_t, y_t = actionPredictionStep(h_t, a_t)
    h_tp1, z_tp1 = observationStep(k_t, o_t)
    return h_tp1, k_t, y_t, z_tp1

# There should always be one more observation than there are actions.
# The goal of this model is to observe something, take an action, and
# predict the next observation. So in order to use the scan function
# we will take care of the first observation here and then append it
# to the results of the scan function
h0, z0 = observationStep(k0, o[0])

[h, k, y, z], _ = theano.scan(step,
    sequences=[a, o[1:]],
    outputs_info=[h0, None, None, None])

# append h0 and z0
T.join(0, h0[numpy.newaxis,:], h)
T.join(0, z0[numpy.newaxis,:], z)

# compare the predictions to the observations. There should be one less
# prediction than there are observations.
loss = T.mean(T.nnet.binary_crossentropy(y, o[1:]))

# compile the theano functions
predict = theano.function(inputs=[o, a], outputs=y)
cost = theano.function(inputs=[o, a], outputs=loss)

# create some fake data. We have 10 samples of 51 observations with 50 actions
observations = numpy.random.uniform(size=(10, 51, 6), low=-.01, high=.01)
actions = numpy.random.uniform(size=(10, 50, 5), low=-.01, high=.01)

# predict should give an error
predict(observations[0], actions[0])

# cost(observations[0], actions[0])
