from dice.die import *
from dice.set1 import *
from dice.solver import *
from dice.printer import Printer
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

correctDieIndex = names.index(randomDie.name)

printer = Printer(numDice, correctDieIndex)

observation = randomDie.obs()
diceProb, expectedEntropy = solver.observe(observation)
obsIndex = featureIndex(observation)

printer.observation(obsIndex, diceProb, expectedEntropy)

# print what feature was observes
# print what expected next features to observe
# print what action was taken
while max(diceProb) != 1.0:
    actionIndex = minIndex(expectedEntropy)
    action = allActions[actionIndex]
    probNext = solver.act(action)

    printer.action(actionIndex, probNext)

    randomDie.act(action)
    observation = randomDie.obs()
    diceProb, expectedEntropy = solver.observe(observation)
    obsIndex = featureIndex(observation)

    printer.observation(obsIndex, diceProb, expectedEntropy)
