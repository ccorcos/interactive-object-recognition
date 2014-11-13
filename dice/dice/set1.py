from dice.die import *
import cPickle as pickle

# So lets start with a set of dice.
#        [f,b,l,r,u,d]
d       = Die([1,2,3,4,5,6], name="d")
dMirror = Die([1,2,3,4,6,5], name="dMirror") # mirror image of d with up-down flipped
dBack   = Die([1,6,3,4,5,2], name="dBack") # back-down flipped
d35Side = Die([1,2,3,5,4,6], name="d35Side") # sides swapped
d36Side = Die([1,2,3,6,4,5], name="d36Side") # sides swapped again

dice = [d, dMirror, dBack, d35Side, d36Side]
names = map(lambda die: die.name, dice)
