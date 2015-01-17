import cPickle as pickle
import numpy
import theano


def getData(name):

    # with open('../../datasets/one-die-optimal.pickle', 'rb') as handle:
    with open('../../datasets/'+name, 'rb') as handle:
        samples = pickle.load(handle)

    # sample = {
    #     'name':die.name,
    #     'note':die.note,
    #     'observations': observations,
    #     'actions':actions,
    #     'diceProbs':diceProbs,
    #     'expectedEntropies':expectedEntropies,
    #     'nextProbs':nextProbs
    # }

    # nextProbs is the correct likelihood of the next feature. This will be used to
    # guage how the modeled learned, but it won't be used to train the model.

    inputs = []
    targets = []
    nextProbs = []
    for sample in samples:
        i = []
        o = []
        p = []
        actions = sample['actions']
        observations = sample['observations']
        nextProb = sample['nextProbs']
        for t in range(len(observations)-1):
            i.append(actions[t]+observations[t])
            o.append(observations[t+1])
            p.append(nextProb[t])
        inputs.append(i)
        targets.append(o)
        nextProbs.append(p)

    inputs = numpy.array(inputs, dtype=theano.config.floatX)
    targets = numpy.array(targets, dtype=theano.config.floatX)
    nextProbs = numpy.array(nextProbs, dtype=theano.config.floatX)

    return inputs, targets, nextProbs