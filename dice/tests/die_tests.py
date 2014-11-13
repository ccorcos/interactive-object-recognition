from unittest import TestCase
import numpy
from se3 import *
from dice.die import *
from dice.set1 import *

pi = numpy.pi


class TestDie(TestCase):
    def test_sides(self):

        labels = ['x','y','z','-x','-y','-z']
        R0 = rpy2R(0.1,0.01,0)
        R1 = rpy2R(0,-pi/2,0.01)

        self.assertTrue(whichAxisIs(R0, facing['front'], labels) == 'x')
        self.assertTrue(whichAxisIs(R0, facing['back'], labels) == '-x')
        self.assertTrue(whichAxisIs(R0, facing['up'], labels) == 'z')
        self.assertTrue(whichAxisIs(R0, facing['down'], labels) == '-z')
        self.assertTrue(whichAxisIs(R0, facing['left'], labels) == '-y')
        self.assertTrue(whichAxisIs(R0, facing['right'], labels) == 'y')

        self.assertTrue(whichAxisIs(R1, facing['front'], labels) == '-z')
        self.assertTrue(whichAxisIs(R1, facing['back'], labels) == 'z')
        self.assertTrue(whichAxisIs(R1, facing['up'], labels) == 'x')
        self.assertTrue(whichAxisIs(R1, facing['down'], labels) == '-x')
        self.assertTrue(whichAxisIs(R1, facing['left'], labels) == '-y')
        self.assertTrue(whichAxisIs(R1, facing['right'], labels) == 'y')

    def test_generateAllPoses(self):
        possibilities = generateAllDiePoses(dice)
        errors = doubleCheckUniqueDiePoses(possibilities)
        self.assertTrue(errors == [])
