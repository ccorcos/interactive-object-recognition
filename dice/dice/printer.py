from dice.die import *
from dice.solver import maxEntropy
from sparkprob import sparkprob


def mark(index, length, marker="*"):
    marked = list(' '*length)
    marked[index] = marker
    return ' '.join(marked)

class Printer:
    def __init__(self, numDice, correctDieIndex):
        """
        Takes the number of dice and the correct dice index.
        """
        self.numDice = numDice
        # print the header
        self.space = ' | '
        self.col1 = numFeatures*2-1
        self.col2 = self.numDice*2-1
        self.col3 = numActions*2-1
        print 'features'.center(self.col1) + self.space + 'dice'.center(self.col2) + self.space + 'actions'.center(self.col3)
        print ' '*self.col1 + self.space + mark(correctDieIndex, self.numDice) + self.space
        self.divider()

    def divider(self):
        print '-'*(self.col1+self.col2+self.col3+len(self.space)*2)

    def observation(self, observationIndex, diceProb, expectedEntropy):
        print mark(observationIndex, numFeatures, marker='x') + self.space + sparkprob(diceProb) + self.space + sparkprob(expectedEntropy, maximum=maxEntropy(self.numDice*self.numDice*numFeatures))
        self.divider()

    def action(self, actionIndex, probNext):
        print sparkprob(probNext) + self.space + ' '*self.col2 + self.space + mark(actionIndex, numActions, marker='x')
        self.divider()
