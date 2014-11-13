from dice.die import *
from dice.set1 import *
from dice.solver import *
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

possibilities, probDice, probNextObservation, expectedDiceEntropyPerAction = solve(dice, observations, actions)

# summarize
correct = names.index(randomDie.name)
space = '    '
print 'p(dice)'.center(len(dice)*2-1) + space + 'E(ent)'.center(len(allActions)*2-1)
star = list(' '*len(dice))
star[correct] = "*"
print ' '.join(star)
for i in range(len(observations)):
    print sparkprob(probDice[i]) + space + sparkprob(expectedDiceEntropyPerAction[i], maximum=maxEntropy(len(dice)*24))
