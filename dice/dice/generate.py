from dice.die import *
from dice.solver import *
import random
import cPickle as pickle

def genSample(die, dice, numActions=8, optimal=True):
    # Model example
    solver = Solver(dice)

    observations = []
    actions = []
    diceProbs  = []
    expectedEntropies = []
    nextProbs = []

    observation = die.obs()
    diceProb, expectedEntropy = solver.observe(observation)

    observations.append(oneHot(observation, allFeatures))
    diceProbs.append(diceProb)
    expectedEntropies.append(expectedEntropy)

    for i in range(numActions):
        action = ''

        if optimal:
            actionIndex = minIndex(expectedEntropy)
            action = allActions[actionIndex]
        else:
            action = random.sample(allActions, 1)[0]

        nextProb = solver.act(action)

        actions.append(oneHot(action,allActions))
        nextProbs.append(nextProb)

        die.act(action)
        observation = die.obs()
        diceProb, expectedEntropy = solver.observe(observation)

        observations.append(oneHot(observation, allFeatures))
        diceProbs.append(diceProb)
        expectedEntropies.append(expectedEntropy)


    sample = {
        'name':die.name,
        'note':die.note,
        'observations': observations,
        'actions':actions,
        'diceProbs':diceProbs,
        'expectedEntropies':expectedEntropies,
        'nextProbs':nextProbs
    }

    return sample

def createDataset2(dice, filename=None, numActions=8, optimal=True, trials=100):
    """
    Parameters:
        dice - an array of Die objects
        filename - an optional filename to save the pickled dataset
        numActions - an optional parameters to specify the number of actions. default: 8
        optional - whether the actions should be optional for object recognition. default: True
    """
    samples = []

    allDice = generateAllDiePoses(dice)
    for trial in range(trials):
        die = random.sample(allDice, 1)[0]
        sample = genSample(die, dice, numActions=numActions, optimal=optimal)
        samples.append(sample)

    if filename:
        with open(filename, 'wb') as handle:
          pickle.dump(samples, handle)

    return samples

def createDataset(dice, filename=None, numActions=8, optimal=True):
    """
    Parameters:
        dice - an array of Die objects
        filename - an optional filename to save the pickled dataset
        numActions - an optional parameters to specify the number of actions. default: 8
        optional - whether the actions should be optional for object recognition. default: True
    """
    samples = []

    allDice = generateAllDiePoses(dice)
    for die in allDice:
        sample = genSample(die, dice, numActions=numActions, optimal=optimal)
        samples.append(sample)

    if filename:
        with open(filename, 'wb') as handle:
          pickle.dump(samples, handle)

    return samples

def loadDataset(filename):

    with open(filename, 'rb') as handle:
      samples = pickle.load(handle)

    return samples
