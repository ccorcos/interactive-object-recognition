from se3 import *
import numpy

pi = numpy.pi

allActions = ['right','left','up','down','back']
allFeatures = [1,2,3,4,5,6]

facing = {'front': numpy.array([1,0,0]), \
          'back': numpy.array([-1,0,0]), \
          'up': numpy.array([0,0,1]),    \
          'down': numpy.array([0,0,-1]), \
          'left': numpy.array([0,-1,0]), \
          'right': numpy.array([0,1,0])}

# find which axis of a rotation matrix, R is closest
# to a specific direction. Returns a label according to
# the labels that correspond to ['x','y','z','-x','-y','-z']
def whichAxisIs(R, direction, labels):
    # the x, y, and z unit vectors are the columns
    positiveXyz = R.T
    # the -x, -y, and -z unit vectors
    negativeXyz = -1*R.T
    # both the positive and negative vectors
    unitVectors = numpy.concatenate((positiveXyz, negativeXyz))
    # projection
    distances = map(lambda x: numpy.dot(x,direction), unitVectors)
    return labels[distances.index(max(distances))]


def oneHot(string, array):
    i = array.index(string)
    oh = [0]*len(array)
    oh[i] = 1
    return oh

def notHot(oh, array):
    i = oh.index(1)
    return array[i]

class Die:
    # side = [0:front, 1:back, 2:left, 3:right, 4:up, 5:down]
    def __init__(self, sides, se3=None, name='', note=''):
        self.sides = sides
        self.name = name
        self.note = note
        if se3:
            self.se3 = se3
        else:
            self.se3 = SE3()

    def whichAxisIs(self, direction):
        direction = facing[direction]
        labels = ['x','y','z','-x','-y','-z']
        R = self.se3.T[0:3,0:3]
        return whichAxisIs(R,direction,labels)

    def whichSideIs(self, direction):
        direction = facing[direction]
        # ['x','y','z','-x','-y','-z'] is [front, right, top, back, left, bottom]
        reorder=[0,3,4,1,2,5]
        labels = [self.sides[i] for i in reorder]
        R = self.se3.T[0:3,0:3]
        return whichAxisIs(R,direction,labels)

    def obs(self):
        return self.whichSideIs('front')

    def observe(self):
        return self.obs()

    def act(self, action):
        return self.turn(action)

    # turn to see a different face
    def turn(self,direction):
        if direction != 'stay':
            directions = ['right', 'left', 'up', 'down','back']
            rotations = [numpy.array([0,0,-pi/2]), numpy.array([0,0,pi/2]), numpy.array([0,pi/2,0]), numpy.array([0,-pi/2,0]), numpy.array([0,0,pi])]
            rotation = rotations[directions.index(direction)]
            self.se3.gRotate(rotation)

    # spin the current face
    def spin(self,direction):
        if direction != 'stay':
            directions = ['right', 'left', 'over']
            rotations = [numpy.array([-pi/2,0,0]), numpy.array([pi/2,0,0]), numpy.array([pi,0,0])]
            rotation = rotations[directions.index(direction)]
            self.se3.gRotate(rotation)

    def fblrud(self):
        return self.whichSideIs('front'), \
               self.whichSideIs('back'),  \
               self.whichSideIs('left'),  \
               self.whichSideIs('right'), \
               self.whichSideIs('up'),    \
               self.whichSideIs('down')

    def __repr__(self):
        # canonical arrangement
        cf,cb,cl,cr,cu,cd = self.sides
        # current arrangement
        f,b,l,r,u,d = self.fblrud()
        paddedName = self.name.center(13)
        return """
%s
     ---
    | %s |
 --- --- ---
| %s | %s | %s |
 --- --- ---
    | %s |
     ---
    | %s |
     ---
""" % (paddedName,u,l,f,r,d,b)

    def __str__(self):
        return "Die(" + str(self.sides) + ", " + str(self.se3) + ", name='" + self.name  + "', note='" + self.note + "')"

    def copy(self):
        return Die(self.sides, SE3(*T2xyzrpy(self.se3.T)), name=self.name, note=self.note)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise
        return numpy.all(self.sides == other.sides)

    def __ne__(self, other):
        return not self.__eq__(other)

    def setFrontLabel(self, label):
        ax = self.whichAxisIs('front')
        i = ['x', '-x', '-y', 'y', 'z', '-z'].index(ax)
        self.sides[i] = label


def printDiceSetRow(dice):
    strings = []
    width = 15
    for die in dice:
        # current arrangement
        paddedName = die.name.center(width)
        f,b,l,r,u,d = die.fblrud()
        strings.append("""
%s
---
| %s |
--- --- ---
| %s | %s | %s |
--- --- ---
| %s |
---
| %s |
---
""" % (paddedName,u,l,f,r,d,b))
    cols = [map(lambda str: str.center(width), string.split('\n')) for string in strings]
    rows = map(list, zip(*cols))
    print '\n'.join([' '.join(row) for row in rows])

def printDiceSet(dice, cols=5):
    chunks = int(numpy.ceil(float(len(dice))/float(cols)))
    groups = [dice[i*cols:(i+1)*cols] for i in range(chunks)]
    for group in groups:
        printDiceSetRow(group)

# Iterate through all poses of all objects
def generateAllDiePoses(dice):
    allDiePoses = []
    for die in dice:
        for spin in ['stay', 'right','left','over']:
            dieSpin = die.copy()
            dieSpin.spin(spin)
            dieSpin.note = spin
            for turn in ['stay','left','right','up','down','back']:
                dieSpinTurn = dieSpin.copy()
                dieSpinTurn.turn(turn)
                dieSpinTurn.note = dieSpinTurn.note + ',' + turn
                allDiePoses.append(dieSpinTurn)
    return allDiePoses

# check that all are different
def doubleCheckUniqueDiePoses(allDiePoses):
    errors = []
    allSides = []
    for i in range(len(allDiePoses)):
        die = allDiePoses[i]
        s = die.fblrud()
        for j in range(len(allSides)):
            side = allSides[j]
            if side == s:
                # print 'we have a match!'
                # print side, s
                # print i,j
                # print allDiePoses[i].note, allDiePoses[j].note
                # printDiceSet([allDiePoses[i], allDiePoses[j]])
                errors.append([allDiePoses[i], allDiePoses[j]])
        allSides.append(s)
    return errors
