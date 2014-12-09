
import numpy
from rnn import *
import matplotlib.pyplot as plt
plt.ion()


print """
Simple lag test to see if the RNN is working properly
"""
n = 20
nin = 5
nout = 2

steps = 10
nseq = 100

numpy.random.seed(0)
# simple lag test
seq = numpy.random.randn(nseq, steps, nin)
targets = numpy.zeros((nseq, steps, nout))

# whether lag 1 (dim 3) is greater than lag 2 (dim 0)
targets[:, 2:, 0] = numpy.cast[numpy.int](seq[:, 1:-1, 3] > seq[:, :-2, 0])

# whether product of lag 1 (dim 4) and lag 1 (dim 2)
# is less than lag 2 (dim 0)
targets[:, 2:, 1] = numpy.cast[numpy.int]((seq[:, 1:-1, 4] * seq[:, 1:-1, 2]) > seq[:, :-2, 0])


rnn = RNN(
    n = n,
    nin = nin,
    nout = nout,
    L1_reg = 0,
    L2_reg = 0
)

rnn.trainModel(
    inputs=seq,
    targets=targets,
    learningRate=0.1,
    epochs=1000
)

seqs = xrange(10)

for seq_num in seqs:
    fig = plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(seq[seq_num])
    ax1.set_title('input')
    ax2 = plt.subplot(212)
    true_targets = plt.step(xrange(steps), targets[seq_num], marker='o')

    guess = rnn.predict(seq[seq_num])
    guessed_targets = plt.step(xrange(steps), guess)
    plt.setp(guessed_targets, linestyle='--', marker='d')
    for i, x in enumerate(guessed_targets):
        x.set_color(true_targets[i].get_color())
    ax2.set_ylim((-0.1, 1.1))
    ax2.set_title('solid: true output, dashed: model output (prob)')

plt.show()
