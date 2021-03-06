##CNN from scratch
## there is an assumption that each example is always 3D tensor (RGB or greyscale)

#### Features :
"""
   -- Different etas for each layer
   -- Different pooling , activations functions per layer and per feature map
   -- b2b convolutional layers (no pooling in between)
   -- pooling Size is assumed to divide the prev layer
   -- You can not have a fully connected proceed a ConvLayer
   -- slowly increasing the batchsize 
   Part of the reason the code may not work is because of numpy stuff
"""

#### Todos :
"""
  -- Add Striding (quickens computation, really ez add-on)
  
  
   """

### if the activation is softmax, then cost is loglikelihood

import random
import numpy as np
import math

class sigmoid:
    @staticmethod
    def func(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def prime(z):
        return sigmoid.func(z) * (1.0 - sigmoid.func(z))



class tanh:
    @staticmethod
    def func(z):
        return np.tanh(z)

    @staticmethod
    def prime(z):
        return 1 - np.square(tanh.func(z))

class arctan:
    @staticmethod
    def func(z):
        return np.arctan(z)

    @staticmethod
    def prime(z):
        return 1 / (np.square(z) + 1)


class ReLu:
    @staticmethod
    def func(z):
        return np.maximum(z, 0)

    @staticmethod
    def prime(z):
        truths = z == np.maximum(z, 0)
        return truths * 1



class ReLuTanh:
    @staticmethod
    def func(z):
        return np.maximum(tanh.func(z), 0)

    @staticmethod
    def prime(z):
        k = tanh.func(z)
        kprime = tanh.prime(z)
        truths = k == np.maximum(k, 0)
        return truths * kprime


class LeakyReLu:
    @staticmethod
    def func(z):
        return np.maximum(z, 0.5 * z) ### This 0.5 could be made into a factor

    @staticmethod
    def prime(z):
        return 1 if z > 0 else 0.5


class softmax:
    @staticmethod
    def func(z):

        exp = np.exp(z)
        theVector = np.sum(exp, axis=0)
        activations = exp / theVector

        return activations


####Cost functions


class loglikedlihood:
    @staticmethod
    def fn(a, y):
        theKey = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(10, 1)
        a = -1 * np.log(a)
        return a * np.matmul(np.transpose(y), theKey)

    @staticmethod
    def delta(z, a, y):
        return a - y


class QuadraticCost:
    @staticmethod
    def fn(a, y):

        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):

        return (a - y) * sigmoid.prime(z)


class CrossEntropyCost:
    @staticmethod
    def fn(a, y):

        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):

        return a - y


## A class for a tpyical convolution (not tiled nor unshared) with optional pooling and striding ;; No padding allowed
class ConvPoolLayer:

    ## assumption that the kernel is less than the input in size
    def __init__(
        self,
        PrevlayerSize,
        KernelSize,
        activationFunctions,
        stride=(1,1),
        PoolingBool=0,
        PoolingMapsFuncs=None,
        PoolingSize=(1, 1, 1)
    ):
        """PrevlayerSize is the size of the previous layer (3D Tensor),
        KernelSize is a tuple of the size of the Kernel (including the number of Channels it convolves with)
        activationFunctions is a list of activation for the convolutional
        if the layer contains Pooling or not and if it does
        PoolingMaps is a list of the pooling function for each layer (function has to return a scalar such as np.max() )
        PoolingSize is a tuple of the 2D pool receptive field  (eg: (2,2))
        self.PoolingLayerSize is size of the pooling layer (the map and the number of layers)
        """
        if KernelSize[-1] != PrevlayerSize[-1]:
            raise Exception("Kernel is not the right size")

        self.PrevlayerSize = PrevlayerSize
        self.ConvolutionalLayerSize = (
            math.floor( ( PrevlayerSize[0] - KernelSize[0] + 1 )     / stride[0] ),
            math.floor(  ( PrevlayerSize[1] - KernelSize[1] + 1 )    / stride[1] ),
            len(PoolingMapsFuncs),
        )
        self.stride = stride
        self.PoolingBool = PoolingBool
        self.PoolingMapsFuncs = PoolingMapsFuncs
        self.PoolingSize = PoolingSize
        c = self.ConvolutionalLayerSize
        self.PoolingLayerSize = (
            int(c[0] / PoolingSize[0]),
            int(c[1] / PoolingSize[1]),
            len(PoolingMapsFuncs),
        )
        
        self.activationFuncs = activationFunctions
        

        ## Will store the weighted input for the last forward pass call
        self.weightedInputs = None
        self.activation = None
        self.activations = None
        self.errors = None
        self.delAdelA = None

        ### Init some Kernels
        self.kernelSize = KernelSize + (len(PoolingMapsFuncs),)
        p = self.kernelSize
        
        self.weights = np.random.normal( loc=1.0, scale=0.2 , size=(p[0], p[1], p[2], p[3])  )   *  0.0005
        self.biases =  np.random.normal( loc=1.0, scale=0.2 , size=(len(PoolingMapsFuncs),1) )   *  0.0005

    def poolingMap(self, batchActivations):
        poolingMaps = self.PoolingMapsFuncs
        listOfMapsByPool = (
            []
        )  ## In this case a map is a 4D tensor of a kernel map for the given batch
        # sizeOfPrev = batchActivations.shape[0:-1]
        listOfdelAdelAByPool = []

        ## iterating over each kernel

        for pool in poolingMaps:
            poolIndex = poolingMaps.index(pool)
            listOfMapsByImage = []
            listOfdelAdelAByImage = []
            poolingFunction = pool
            poolX = self.PoolingSize[0]
            poolY = self.PoolingSize[1]  ###
            ###this is potentially

            for inputCount in range(batchActivations.shape[-1]):
                input = batchActivations[:, :, poolIndex - 1, inputCount]
                map = np.zeros((self.PoolingLayerSize[0], self.PoolingLayerSize[1], 1, 1))
                delAdelA = None

                ## This is with assumption that no overlapping is done in pooling
                listOfdelAdelAByNeuronX = []
                for x in range(self.PoolingLayerSize[0]):
                    listOfdelAdelAByNeuronY = []
                    for y in range(self.PoolingLayerSize[1]):
                        map[x][y][0][0] = poolingFunction(
                            input[2 * x : 2 * x + poolX, 2 * y : 2 * y + poolY]
                        )

                        ### Lets make the pooling function just maxpooling for now , L2/L1 average is quite ez
                        delAdelA = (
                            input[2 * x : 2 * x + poolX, 2 * y : 2 * y + poolY]
                            == map[x][y][0][0]
                        )[:, :]
                        listOfdelAdelAByNeuronY.append(delAdelA)
                    listOfdelAdelAByNeuronX.append(
                        np.concatenate(listOfdelAdelAByNeuronY, axis=1)
                    )

                listOfMapsByImage.append(map)
                listOfdelAdelAByImage.append(
                    np.concatenate(listOfdelAdelAByNeuronX, axis=0)[
                        :, :, np.newaxis, np.newaxis
                    ]
                )
            listOfdelAdelAByPool.append(np.concatenate(listOfdelAdelAByImage, axis=3))
            listOfMapsByPool.append(np.concatenate(listOfMapsByImage, axis=3))
            

        ### This may fail
        self.delAdelA = np.concatenate( listOfdelAdelAByPool , axis=2 )

        return np.concatenate(listOfMapsByPool, axis=2)

    def feedForward(self, inputActivation):

        ## input Activation is guranteed to be 4D tensor; for eg (5,8,3,10)  where 10 is the m
        kw = self.weights  ## the 4D Tensor, weights  (5,8,3,2)
        kw = kw[:, :, :, :, np.newaxis]  ##(5,8,3,2,1)
        kb = self.biases  ## the 1D vector, biases  (2,)
        ks = self.kernelSize + (1,) ## the size of weights tuple 

        # sizeOfPrev = inputActivation.shape[0:-1]

        listOfXWeightedInput = []

        for x in range(self.ConvolutionalLayerSize[0]):

            listOfYWeightedInput = []

            for y in range(self.ConvolutionalLayerSize[1]):
                qarray = inputActivation[
                    x :  ( x * self.stride[0] + ks[0])  , y : (y * self.stride[1] + ks[1]), :, np.newaxis,:
                ]  ### (5,8,3,1,10)

                ### Convolution with double broadcasting
                doubleBroadcastArray = qarray * kw 
                for number in range(0, 3):
                    doubleBroadcastArray = np.sum(
                        doubleBroadcastArray, axis=0
                    )  ### we get two dimensional, (2,10)
                doubleBroadcastArray = doubleBroadcastArray + kb
                # xyneuronWeightedInput = np.flip(
                #     np.rot90(doubleBroadcastArray, k=1, axes=(0, 1)), axis=0
                # )
                xyneuronWeightedInput = doubleBroadcastArray[
                    np.newaxis, np.newaxis, :, :
                ]  ## 4D Tensor (1,1,2,10)
                # xyActivation = self.activationFuncs.func(xyneuronWeightedInput)

                listOfYWeightedInput.append(xyneuronWeightedInput)

            listOfXWeightedInput.append(np.concatenate(listOfYWeightedInput, axis=1))

        self.weightedInputs = np.concatenate(listOfXWeightedInput, axis=0)
        listofActiv = []
        
        for index in range(self.weightedInputs.shape[-2]):
            listofActiv.append( self.activationFuncs[index].func(
                self.weightedInputs[:, :, index, np.newaxis, :]
            ) )

        self.activation = np.concatenate(listofActiv, axis=2)
        # print(self.activation.shape)
        

        if self.PoolingBool == 1:
            k = self.poolingMap(self.activation)
            self.activations = k
        else:
            self.activations = self.activation

    def backprop(
        self, eta, batchSize, batchLabels, inputActivation, deeperDeltas, deeperWeights
    ):
        ### BP1 does not apply to Conv Pool Layer

        c = self.ConvolutionalLayerSize

        ### BP 2
        if self.PoolingBool == 1:

            deeperWeights = (deeperWeights.transpose()).reshape(
                (
                    self.PoolingLayerSize[0],
                    self.PoolingLayerSize[1],
                    self.PoolingLayerSize[2],
                    deeperWeights.shape[0],
                ),
                order="F",
            )

            deeperWeights = deeperWeights[:, :, :, :, np.newaxis]  ### (12,12,3,10,1)
            deeperDeltas = deeperDeltas[
                np.newaxis, np.newaxis, np.newaxis
            ]  ### (1,1,1,10,m)
            doubleBroadcastArray = deeperWeights * deeperDeltas  ## (12,12,3,10,m)
            deltasTimesWeights = np.sum(
                doubleBroadcastArray, axis=3
            )  ## (12,12,3,m) so there are m of sum(delta*weight) for each neuron in the layer
            s = deltasTimesWeights.shape
            deltasTimesWeightsAugmented = np.kron(
                deltasTimesWeights,
                np.ones((self.PoolingSize[0], self.PoolingSize[1]) + (1, 1)),
            )

            # self.errors = np.zeros((c[0], c[1]) + (c[2], batchSize))

            # for index in range(c[2]):
            #     self.errors[:, :, index, :] = (
            #         self.delAdelA[:, :, index, :]
            #         * 
            #         self.activationFuncs[index].prime(
            #             self.weightedInputs[:, :, index, :]
            #         )
            #         * deltasTimesWeightsAugmented[:, :, index, :]
            #     )

            
                        
            print(self.weightedInputs.shape)
            print(deltasTimesWeightsAugmented.shape)
            
            self.errors = (
                    self.delAdelA
                    * 
                    self.activationFuncs[0].prime(
                        self.weightedInputs
                    )
                    * deltasTimesWeightsAugmented
                )
            
            
        else:
            ## There is no pooling or downsampling that happens on the convolutional layer

            ## If The deeper layer is Fully Connected
            if len(deeperDeltas.shape) == 2:
                deeperWeights = (deeperWeights.transpose()).reshape(
                    (c[0], c[1], c[2], deeperWeights.shape[0]), order="F"
                )

                deeperWeights = deeperWeights[
                    :, :, :, :, np.newaxis
                ]  ### (24,24,3,10,1)
                deeperDeltas = deeperDeltas[
                    np.newaxis, np.newaxis, np.newaxis
                ]  ### (1,1,1,10,m)
                doubleBroadcastArray = deeperWeights * deeperDeltas  ## (24,24,3,10,m)
                deltasTimesWeights = np.sum(
                    doubleBroadcastArray, axis=3
                )  ## (12,12,3,m) so there are m of sum(delta*weight) for each neuron in the layer
                self.errors = np.zeros((c[0], c[1]) + (c[2], batchSize))

                # print(self.weightedInputs.shape , deltasTimesWeights.shape)
                
                # for index in range(c[2]):
                #     if( deltasTimesWeights.shape[-1] ==  2  ):
                #         print(self.activationFuncs[index].prime(
                #             self.weightedInputs[:, :, index, :]
                #         ).shape)
                        
                        
                self.errors = self.activationFuncs[0].prime( self.weightedInputs) * deltasTimesWeights
                    

            ## The deeper layer is Convolutional Layer
            else:
                pass

        ## BP3 ; updating the Biases
        self.biases = self.biases - ((eta) / batchSize) * np.sum(
            np.sum(self.errors, axis=-1), axis=(0, 1)
        )[:,np.newaxis]

        iaR = inputActivation[:, :, :, np.newaxis, :]
        # iaRslide = np.lib.stride_tricks.as_strided(
        #     iaR, shape=(self.kernelsize[0] , self.kernelsize[1]) + (c[0], c[1]) + iaR.shape[2:]  ### ofc the shape of c is less than that of input activation
        # , stride=self.stride)
        iaRslide = np.lib.stride_tricks.sliding_window_view(
            iaR, (c[0], c[1]) + iaR.shape[2:]  ### ofc the shape of c is less than that of input activation
        ) [ ::self.stride[0] , ::self.stride[1] ]
        
        iaRslide = iaRslide.sum(axis=(2, 3, 4))
        e = self.errors[np.newaxis, np.newaxis, :, :, np.newaxis, :, :]

        
        
        
        # print(self.weights.shape, iaRslide.shape, e.shape)
        ## BP4 : updating the weights
        self.weights = self.weights - ((eta) / batchSize) * (
            (iaRslide * e).sum(axis=(2, 3, -1))
        )

        # (5, 5, 1, 3) (5, 5, 24, 24, 1, 1, 10) (1, 1, 24, 24, 1, 3, 10) -> 5, 5,  1, 3,


class FullyConnected:
    def __init__(self, PrevlayerSize, size, activationFunction, costFunction=None):
        """PrevlayerSize is a tuple of the previous layer,
        size of the layer as 1D tuple"""

        self.PrevlayerSize = 1
        for k in PrevlayerSize:
            self.PrevlayerSize *= k

        self.weights = np.random.normal(loc=1.0, scale=0.2, size=(size[0], self.PrevlayerSize)) * 0.0005  ## 
        self.biases = np.random.normal(loc=1.0, scale=0.2, size=(size[0], 1))  * 0.0005
        self.activationFunc = activationFunction

        ###Forbackprop
        self.activations = None
        self.weightedInputs = None
        self.errors = None
        self.costFunction = costFunction

    def feedForward(self, inputActivation):

        ## This works for Convolutional or Fully Connected layer
        # print(inputActivation.shape)
        inputActivation = inputActivation.reshape(
            (self.PrevlayerSize, inputActivation.shape[-1]), order="F"
        )
        z = np.matmul(self.weights, inputActivation) + self.biases
        self.weightedInputs = z
        self.activations = self.activationFunc.func(z)

    def backprop(
        self, eta, batchSize, batchLabels, inputActivation, deeperDeltas, deeperWeights
    ):
        inputActivation = inputActivation.reshape(
            (self.PrevlayerSize, inputActivation.shape[-1]), order="F"
        )
        NablaW = None
        Deltas = None
        NablaB = Deltas

        #### BP1 / BP2 ########
        ## BP2
        if self.costFunction == None:

            Deltas = (
                np.matmul(np.transpose(deeperWeights), deeperDeltas)
            ) * self.activationFunc.prime(self.weightedInputs)

        else:
            ##BP1  : this is for all cost and all activations incl. softmax

            Deltas = self.costFunction.delta(
                self.weightedInputs, self.activations, batchLabels
            )

            ##BP3/4

        self.errors = Deltas
        NablaW = (
            np.rot90(inputActivation[:, :, np.newaxis], k=1, axes=(2, 0))
            * Deltas[:, :, np.newaxis]
        )
        NablaW = np.sum(NablaW, axis=1)
        # print(NablaW.shape, self.weights.shape)
        NablaB = Deltas

        self.weights = self.weights - (eta * NablaW / batchSize)
        self.biases = self.biases - (eta * np.sum(NablaB, axis=1) / batchSize)[:,np.newaxis]


class inputLayer:
    def __init__(self, size):
        self.size = size
        self.activations = None


class Network:

    ## the first layer is an input layer

    def __init__(self, layers, etas, bacthSize=10):
        """etas is a list of etas for each layer
        layers is a list of layers with the first being a tuple (i.e. the dimension of the input)

        """

        ## Sets up the layers of the network
        self.numlayers = len(layers)
        self.layers = layers

        if type(etas) == list:
            if len(etas) != len(layers) - 1:
                raise Exception("Not enough etas")

        self.etas = etas
        self.m = bacthSize
        self.n = 1

        ## backprop to find gradient of all C_x in one batch

    def backprop(self, batch):
        nq = self.numlayers  ## nq  layers w/ the input

        ## Feedforward to find all the activation

        batchImages = np.concatenate(
            [ x[:, :, :, np.newaxis] for x, y in batch], axis=-1
        )  ## 4Dtensor
        batchLabels = np.hstack([y for x, y in batch])
        self.layers[0].activations = batchImages

        ### for each layer, we feed forward
        for layerIndex in range(1, nq):
            layer = self.layers[layerIndex]
            layer.feedForward(self.layers[layerIndex - 1].activations)

        ##########  ***** Back Propogatioon *******#########
        ## The last layer
        #  eta, batchSize, batchLabels, inputActivation, deeperDeltas, deeperWeights

        ((self.layers)[-1]).backprop(
            self.etas[-1],
            self.m,
            batchLabels,
            self.layers[-2].activations,
            deeperDeltas=None,
            deeperWeights=None,
        )

        ##Every other layer
        for layerIndex in range(2, nq):
            self.layers[layerIndex * -1].backprop(
                self.etas[layerIndex * -1],
                self.m,
                None,
                self.layers[(layerIndex * -1) - 1].activations,
                deeperDeltas=self.layers[(layerIndex * -1) + 1].errors,
                deeperWeights=self.layers[(layerIndex * -1) + 1].weights,
            )
            ### This code will break due to kernels in ConvLayers

    def SGD(
        self, batchSize, etas, epochs, trainingSet
    ):  ## the epoch could later be made into a float
        leng = len(trainingSet)

        if leng == 0:
            return
        self.etas = etas
        self.m = batchSize
        m = self.m

        for T in range(0, epochs):
            ## shuffle up the training example and divy them up into batches of the size batchSize
            random.shuffle(trainingSet)
            setOfBatches = [trainingSet[q : q + m] for q in range(0, leng, m)]

            for Batch in setOfBatches:

                # m = math.min(int(math.floor(m * 1.001)) , 500)
                self.backprop(Batch)

## returns the ANN's answer to the input image x
def feedForward( batch, layers):
    nq = 3

    batchImages = np.concatenate(
        [x[:, :, :, np.newaxis] for x, y in batch], axis=-1
    )

    batchLabels = [y for x, y in batch]
    layers[0].activations = batchImages
    # print(batchImages)

    for layerIndex in range(1,nq):
        layer = layers[layerIndex]
        layer.feedForward(layers[layerIndex - 1].activations)
        # print(layer.activations)

    return layers[-1].activations, batchLabels

def eval(testImages, batchSize, layers):  ## measure accuracy
    results = []
    output = []
    for interval in range(0, len(testImages), batchSize):
        section = testImages[interval : interval + batchSize]
        x, y = feedForward(section, layers)
        answers = [np.argmax(x[:, q]) for q in range(len(section))]
        ans = [ x[:,q] for q in range(len(section))]
        answers = zip(answers, y)
        ans = zip(ans, y)
        results += answers
        output += ans

    accuracy = sum( int(x == y) for (x, y) in results)
    return output, results, accuracy, accuracy  / len(testImages)


def save(filename, Arch):
        x = ""
        for layer in Arch:
          x += "\n" + json.dumps(layer.__dict__)

        y = x.encode('utf-8')
        with open(filename, 'w') as f:
          f.write(str(y))

        # drive.mount("\content\drive")
        files.download(filename)

