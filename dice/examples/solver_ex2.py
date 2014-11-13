from dice.die import *
from dice.set1 import *
from dice.solver import *
import random
import numpy
import os
from sparkprob import sparkprob

# lets take some random actions and observations
randomDie = random.choice(generateAllDiePoses(dice)).copy()

def mark(index, length, marker="*"):
    marked = list(' '*length)
    marked[index] = marker
    return ' '.join(marked)

print """
In the 'features' column, 'x' represents an observed feature
and the histogram represented the likelihood of seeing each
feature after the given action is taken.

In the 'dice' column, the '*' represented the correct die and
the histogram represents the likelihood of each die given all
previous observations and actions.

In the 'actions' column, the histogram represents the expected
entropy of object probabilites for each action. The 'x'
represents the action that was taken which is selected by the
minimum expected entropy.
"""
# Model example
solver = Solver(dice)

# printing
space = ' | '
col1 = len(allFeatures)*2-1
col2 = len(dice)*2-1
col3 = len(allActions)*2-1
print 'features'.center(col1) + space + 'dice'.center(col2) + space + 'actions'.center(col3)
correct = names.index(randomDie.name)
print ' '*col1 + space + mark(correct, len(dice)) + space

observation = randomDie.obs()
diceProb, expectedEntropy = solver.observe(observation)
observed = allFeatures.index(observation)

print '-'*(col1+col2+col3+len(space)*2)
print mark(observed, len(allFeatures), marker='x') + space + sparkprob(diceProb) + space + sparkprob(expectedEntropy, maximum=maxEntropy(len(dice)*24))

# print what feature was observes
# print what expected next features to observe
# print what action was taken
while max(diceProb) != 1.0:
    print '-'*(col1+col2+col3+len(space)*2)
    actionIndex = minIndex(expectedEntropy)
    action = allActions[actionIndex]
    probNext = solver.act(action)
    print sparkprob(probNext) + space + ' '*col2 + space + mark(actionIndex, len(allActions), marker='x')
    print '-'*(col1+col2+col3+len(space)*2)
    randomDie.act(action)
    observation = randomDie.obs()
    diceProb, expectedEntropy = solver.observe(observation)
    observed = allFeatures.index(observation)
    print mark(observed, len(allFeatures), marker='x') + space + sparkprob(diceProb) + space + sparkprob(expectedEntropy, maximum=maxEntropy(len(dice)*24))
