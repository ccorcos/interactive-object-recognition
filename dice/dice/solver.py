from dice.die import *
import numpy
import random

def minIndex(arr):
    m = min(arr)
    idxs = [i for i, x in enumerate(arr) if x == m]
    return random.choice(idxs)

def eliminatePossibilities(possibilities, obs):
    possible = []
    for die in possibilities:
        if die.whichSideIs('front') is obs:
            possible.append(die)
    return possible

def probOfEachDie(possibilities, names):
    # count the number of occurances of each name
    c = counts(map(lambda die: die.name, possibilities), names)
    total = float(sum(c))
    probs = map(lambda count: count/total, c)
    return probs

def counts(arr, values):
    count = [0]*len(values)
    for item in arr:
        for idx, val in enumerate(values):
            if item == val:
                count[idx] = count[idx] + 1
    return count

def expectedEntropies(possibilities):
    # This calculates the exepcted entropy for each action across the
    # likelihood of object-poses.
    entropies = []
    for action in allActions:
        nextObservation = []
        for die in possibilities:
            d = die.copy()
            d.turn(action)
            nextObservation.append(d.whichSideIs('front'))
        count = counts(nextObservation, allFeatures)
        ent = []
        for i in count:
            for j in range(i):
                # probability is 1/i for i possible choices
                ent.append(-numpy.log2(1.0/float(i)))
        entropies.append(numpy.mean(ent))
    # bestAction = allActions[entropies.index(min(expectedEntropies))]
    return entropies

def probOfEachFeature(possibilities):
    possibleObservations = map(lambda die: die.whichSideIs('front'), possibilities)
    c = counts(possibleObservations, allFeatures)
    total = float(sum(c))
    probs = map(lambda count: count/total, c)
    return probs

def solve(dice, observations, actions):
    probDice = []
    probNextObservation = []
    expectedDiceEntropyPerAction = []

    names = map(lambda die: die.name, dice)
    possibilities = generateAllDiePoses(dice)

    # eliminate possiblities for just the first observation
    possibilities = eliminatePossibilities(possibilities, observations[0])
    # compute the probability of each dice thats remaining
    probDice.append(probOfEachDie(possibilities, names))
    # compute the expected entropy
    expectedDiceEntropyPerAction.append(expectedEntropies(possibilities))

    for o, a in zip(observations[1:], actions):
        # compute action on each die
        for die in possibilities:
            die.turn(a)
        # check what features are showing
        probNextObservation.append(probOfEachFeature(possibilities))
        # eliminate possiblities
        possibilities = eliminatePossibilities(possibilities, o)
        # compute the probability of the dice that are remaining
        probDice.append(probOfEachDie(possibilities, names))
        # compute the expected entropy
        expectedDiceEntropyPerAction.append(expectedEntropies(possibilities))

    return possibilities, probDice, probNextObservation, expectedDiceEntropyPerAction

# implement solve in a more incremental manner
class Solver:
    def __init__(self, dice):
        self.observations = []
        self.actions = []
        self.probDice = []
        self.probNextObservation = []
        self.expectedDiceEntropyPerAction = []

        self.names = map(lambda die: die.name, dice)
        self.possibilities = generateAllDiePoses(dice)

    def observe(self, observation):
        # check that we are alternating
        if len(self.observations) != len(self.actions): raise
        # remember everything
        self.observations.append(observation)
        # eliminate possiblities
        self.possibilities = eliminatePossibilities(self.possibilities, observation)
        # compute the probability of each dice thats remaining
        prob = probOfEachDie(self.possibilities, self.names)
        self.probDice.append(prob)
        # compute the expected entropy
        entropy = expectedEntropies(self.possibilities)
        self.expectedDiceEntropyPerAction.append(entropy)
        return prob, entropy

    def act(self, action):
        # check that we are alternating
        if len(self.observations) != (len(self.actions)+1): raise
        # remember everything
        self.actions.append(action)
        # compute action on each die
        for die in self.possibilities:
            die.turn(action)
        # check what features are showing
        probNext = probOfEachFeature(self.possibilities)
        self.probNextObservation.append(probNext)
        return probNext

    def done():
        return self.possibilities, self.probDice, self.probNextObservation, self.expectedDiceEntropyPerAction

def maxEntropy(nDice):
    return -numpy.log2(1.0/float(nDice))
