import random
import json
import sys

import numpy as np


## vectorized form of the sigmoid
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoidPrime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


class Network(object):

    ## the first layer is an input layer

    def __init__(self, layerSizes):  ## [3,4,5,6]

        ## Sets up the layers of the network
        self.numlayers = len(layerSizes)
        self.sizes = layerSizes

        ## Sets up the random biases and weights for each layer starting at 2
        self.weights = [
            np.random.randn(layerSizes[x], layerSizes[x - 1]) * 0.00001
            for x in xrange(1, self.numlayers)
        ]

        self.biases = [np.random.randn(n, 1) * 0.00001 for n in self.sizes[1:]]

        self.eta = 0
        self.m = 0

        ## backprop to find gradient of C_x
    def backprop(self, image, label):
        nq = self.numlayers

        ## Feedforward to find all the activation
        activations = [image]
        ArrayofZs = []
        for l in xrange(0, nq - 1):
            Z_l = np.dot(self.weights[l], activations[l]) + self.biases[l]
            ArrayofZs.append(Z_l)

            ## the activation vector for the lth layer
            activationl = sigmoid(Z_l)
            activations.append(activationl)

        arrayOfNablaW = [0 for w_l in self.weights]
        arrayOfNablaB = ArrayofDeltas = [0 for b_l in self.biases]

        ## l = 0
        ## backprop using the error approach
        ## Calculate the error for the output layer L
        ArrayofDeltas[nq - 2] = ((activations[nq - 1] - label) *
                                 sigmoidPrime(ArrayofZs[nq - 2]))

        arrayOfNablaW[nq - 2] = np.dot(ArrayofDeltas[nq - 2],
                                       activations[nq - 2].transpose())

        for l in xrange(1, nq - 1):

            ArrayofDeltas[nq - 2 - l] = (np.dot(
                self.weights[nq - 2 - l + 1].transpose(),
                ArrayofDeltas[nq - 2 - l + 1])) * sigmoidPrime(
                    ArrayofZs[nq - 2 - l])

            arrayOfNablaW[nq - 2 - l] = np.dot(
                ArrayofDeltas[nq - 2 - l], activations[nq - 2 - l].transpose())

        return (arrayOfNablaB, arrayOfNablaW)

    ## the update rule execution for a given mini batch
    def updateParameters(self, batch):

        ## each ith NablaW is a matrix of the corresponding partial derivatives for the i + 2 th layer
        arrayOfNablaW = [np.zeros(w_l.shape) for w_l in self.weights]
        ## each ith NablaB is a vector of the correpsonding partial derivatives for the i+2 th layer
        arrayOfNablaB = [np.zeros(b_l.shape) for b_l in self.biases]

        for image, label in batch:
            qarrayOfNablaB, qarrayOfNablaW = self.backprop(image, label)
            arrayOfNablaW = [
                w_l + qw_l for w_l, qw_l in zip(arrayOfNablaW, qarrayOfNablaW)
            ]
            arrayOfNablaB = [
                b_l + qb_l for b_l, qb_l in zip(arrayOfNablaB, qarrayOfNablaB)
            ]

        ## update the parameters for each mini batch
        # self.weights = [ w_l - (self.eta / self.m) * arrayOfNablaW[self.weights.index(w_l)] for w_l in self.weights]
        # self.biases = [b_l - (self.eta / self.m) * arrayOfNablaB[self.biases.index(b_l)] for b_l in self.biases]
        # (self.eta / np.sqrt(nw.dot(nw)))
        self.weights = [
            w - (self.eta / self.m) * nw
            for w, nw in zip(self.weights, arrayOfNablaW)
        ]
        self.biases = [
            b - (self.eta / self.m) * nb
            for b, nb in zip(self.biases, arrayOfNablaB)
        ]

    def SGD(self, batchSize, eta, epochs,
            trainingSet):  ## the epoch could later be made into a float
        leng = len(trainingSet)
        if leng == 0:
            return
        self.eta = eta
        self.m = batchSize
        m = self.m

        # print([w_l.shape for w_l in self.biases])

        for T in xrange(0, epochs):
            ## shuffle up the training example and divy them up into batches of the size batchSize
            random.shuffle(trainingSet)
            setOfBatches = [trainingSet[q:q + m] for q in xrange(0, leng, m)]
            for Batch in setOfBatches:
                self.updateParameters(Batch)

    ## returns the ANN's answer to the input image x
    def feedForward(self, x):
        answer = x
        for l in xrange(0, self.numlayers - 1):
            answer = sigmoid(np.dot(self.weights[l], answer) + self.biases[l])
        return answer

    def eval(self, testImages):
        answers_loss = [(np.argmax(self.feedForward(Image)), label)
                        for (Image, label) in testImages]

        # print ([((x,y)) for (x, y) in answers_loss])
        # print(len(answers_loss))
        return sum(int(x == y) for (x, y) in answers_loss)

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": str(self.cost.__name__)
        }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
