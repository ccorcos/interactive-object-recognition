from dice.die import *

# [0:front, 1:back, 2:left, 3:right, 4:up, 5:down]
d = Die([1,2,3,4,5,6])
print 'dice starting at [1,2,3,4,5,6]'
print repr(d)
print 'turn right'
d.turn('right')
print repr(d)
print 'turn right again'
d.turn('right')
print repr(d)
print 'turn left'
d.turn('left')
print repr(d)
print 'turn up'
d.turn('up')
print repr(d)
print 'turn right'
d.turn('right')
print repr(d)
print 'turn left'
d.turn('left')
print repr(d)

print 'You can also print out the constructor:'
print str(d)
d2 = eval(str(d))
print repr(d)
print repr(d2)
