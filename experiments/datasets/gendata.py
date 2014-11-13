from dice.die import *
from dice.generate import *
from dice.set1 import *

# create a dataset of just one die. we would expect that a RNN could learn
# this very easily. A HMM should be able to learn this as well, but it would
# be a very nice comparison of the computational complexity, etc.
oneDieSet = [dice[0]]

# _ = createDataset(oneDieSet, filename='one-die-not-optimal.pickle', numActions=50, optimal=False)
# _ = createDataset(oneDieSet, filename='one-die-optimal.pickle', numActions=50, optimal=True)

# _ = createDataset(dice, filename='set1-not-optimal-50.pickle', numActions=50, optimal=False)
# _ = createDataset(dice, filename='set1-die-optimal-8.pickle', numActions=8, optimal=True)
