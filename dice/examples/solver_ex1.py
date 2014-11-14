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

# solve example
# print 'Generate a random walk
def walk(die):
    die = die.copy()
    observations = []
    actions = []
    observations.append(die.obs())
    # while you havent visited all sides at least once
    while (not numpy.all(map(lambda x: observations.count(x) >= 1, allFeatures))):
        action = random.choice(allActions)
        actions.append(action)
        die.act(action)
        observations.append(die.obs())
    return observations, actions

observations, actions = walk(randomDie)

possibilities, probDice, probNext, expectedEntropy = solve(dice, observations, actions)

# summarize
correctDieIndex = names.index(randomDie.name)
printer = Printer(numDice, correctDieIndex)

printer.observation(featureIndex(observations[0]), probDice[0], expectedEntropy[0])

for i in range(len(actions)):
    printer.action(actionIndex(actions[i]), probNext[i])

    printer.observation(featureIndex(observations[i+1]), probDice[i+1], expectedEntropy[i+1])
