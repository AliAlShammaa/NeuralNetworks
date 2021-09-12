import random
import numpy as np



## matricized form of the sigmoid
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
            np.random.randn(layerSizes[x], layerSizes[x - 1])  * 0.0001
            for x in xrange(1, self.numlayers)
        ]
        self.weight_velocities = [
            np.zeros(layerSizes[x], layerSizes[x - 1])
            for x in xrange(1, self.numlayers)
        ]
        self.biases = [np.random.randn(n, 1) * 0.0001 for n in self.sizes[1:]]

        self.eta = 0
        self.m = 0
        self.n = 1
        self.mu = 0

        ## backprop to find gradient of C_x
    def backprop(self, batch):
        nq = self.numlayers

        ## Feedforward to find all the activation
        batchImages = np.hstack([x for x, y in batch])
        batchLabels = np.hstack([y for x, y in batch])
        batchActivations = [batchImages]
        batchArrayofZs = []
        for l in xrange(0, nq - 1):
            Z_l = np.matmul(self.weights[l],
                            batchActivations[l]) + self.biases[l]
            batchArrayofZs.append(Z_l)  ## you are adding matrix here
            ## the activation vector for the lth layer
            activationl = sigmoid(Z_l)
            batchActivations.append(activationl)

        arrayOfNablaW = [0 for w_l in self.weights]
        arrayOfNablaB = ArrayofDeltas = [0 for b_l in self.biases]

        ## l = 0
        ## backprop using the error approach
        ## Calculate the error for the output layer L

        ## This is a matrix
        ArrayofDeltas[nq - 2] = ((batchActivations[nq - 1] - batchLabels) *
                                 sigmoidPrime(batchArrayofZs[nq - 2]))


        #
        # print ( np.repeat(
        #     ArrayofDeltas[nq - 2][:, :, np.newaxis],
        #     self.sizes[nq - 2],
        #     axis=2))
        # print("\n \n")
        # print(
        #       np.rot90(np.repeat(batchActivations[nq - 2][:, :,
        #                          np.newaxis],
        #                          self.sizes[nq - 1],
        #                          axis=2),
        #                k=1,
        #                axes=(0, 2)
        #                ))
        # print("\n \n")



        arrayOfNablaW[nq - 2] = np.repeat(
            ArrayofDeltas[nq - 2][:, :, np.newaxis],
            self.sizes[nq - 2],
            axis=2) * np.rot90(np.repeat(batchActivations[nq - 2][:, :,
                                                                  np.newaxis],
                                         self.sizes[nq - 1],
                                         axis=2),
                               k=1,
                               axes=(0, 2))

        arrayOfNablaW[nq - 2] = arrayOfNablaW[nq - 2].sum(axis=1)

        for l in xrange(1, nq - 1):

            ArrayofDeltas[nq - 2 - l] = (np.matmul(
                self.weights[nq - 2 - l + 1].transpose(),
                ArrayofDeltas[nq - 2 - l + 1])) * sigmoidPrime(
                    batchArrayofZs[nq - 2 - l])

            arrayOfNablaW[nq - 2 - l] = np.repeat(
                ArrayofDeltas[nq - 2 - l][:, :, np.newaxis],
                self.sizes[nq - 2 - l],
                axis=2) * np.rot90( np.repeat(
                    batchActivations[nq - 2 - l][:, :, np.newaxis],
                    self.sizes[nq - 2 - l + 1],
                    axis=2),
                                   k=1,
                                   axes=(0, 2))

            arrayOfNablaW[nq - 2 - l] = arrayOfNablaW[nq - 2 - l].sum(axis=1)
            ArrayofDeltas[nq - 1 - l] = ArrayofDeltas[nq - 1 - l].sum(
                axis=1)[:, np.newaxis]

        ArrayofDeltas[0] = ArrayofDeltas[0].sum(axis=1)[:, np.newaxis]

        return (arrayOfNablaB, arrayOfNablaW)



    ## the update rule execution for a given mini batch
    def updateParameters(self, batch):

        ## each ith NablaW is a matrix of the corresponding partial derivatives for the i + 2 th layer
        ## each ith NablaB is a vector of the correpsonding partial derivatives for the i+2 th layer
        arrayOfNablaB, arrayOfNablaW = self.backprop(batch)

        self.weight_velocities = [v - (self.eta / self.m) * nw - self.mu * (1/v)
                                  for v, nw in zip(self.weight_velocities, arrayOfNablaW)]
        # self.bias_velocities = [mu * v - (eta / self.m) * nb
        #                         for v, nb in zip(self.bias_velocities, nabla_b)]



        ## update the parameters for each layer
        self.weights = [
            w - (self.eta / self.m) * nw
            for w, nw in zip(self.weights, self.weight_velocities)
        ]


        self.biases = [
            b - (self.eta / self.m) * nb
            for b, nb in zip(self.biases, arrayOfNablaB)
        ]

    def SGD(self, batchSize, eta, epochs, mu,
            trainingSet):  ## the epoch could later be made into a float
        leng = len(trainingSet)

        if leng == 0:
            return
        self.eta = eta
        self.m = batchSize
        self.mu = mu
        m = self.m

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
        return sum(int(x == y) for (x, y) in answers_loss)
