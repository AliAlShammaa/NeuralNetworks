##CNN from scratch 
## there is an assumption that each example is always 3D tensor (RGB or greyscale)

#### Features :
'''
   -- Different etas for each layer
   -- Different pooling , activations functions per layer and per feature map
   -- b2b convolutional layers (no pooling in between)
   -- pooling Size is assumed to divide the prev layer
   -- You can not have a fully connected proceed a ConvLayer
   -- slowly increasing the batchsize 
   Part of the reason the code may not work is because of numpy stuff
'''

#### Todos :
'''
  -- Add Striding (quickens computation, really ez add-on)
  
  
   '''

### if the activation is softmax, then cost is loglikelihood

import random
import numpy as np


class sigmoid():
  @staticmethod
  def func(z):
      return 1.0 / (1.0 + np.exp(-z))

  @staticmethod
  def prime(z):
      return sigmoid.func(z) * (1.0 - sigmoid.func(z))



class tanh():
  @staticmethod
  def func(z):
      return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

  @staticmethod
  def prime(z):
      return 1 - np.square(tanh.func(z))


class ReLu():
  @staticmethod
  def func(z):
      return z if z > 0 else 0

  @staticmethod
  def prime(z):
      return 1 if z > 0 else 0

class ReLuTanh():
  @staticmethod
  def func(z):
      return tanh.func(z) if z > 0 else 0

  @staticmethod
  def prime(z):
      return tanh.prime(z) if z > 0 else 0

class LeakyReLu():
  @staticmethod
  def func(z):
      return z if z > 0 else (0.5 * z)  ### This 0.5 could be made into a factor

  @staticmethod
  def prime(z):
      return 1 if z > 0 else 0.5

class softmax():

  @staticmethod
  def func(z):

    exp = np.exp(z)
    theVector = np.sum(exp,axis=0)
    activations = exp / theVector

    return activations

####Cost functions

class loglikedlihood():
    
    @staticmethod
    def fn(a, y):
        theKey =  np.array([0,1,2,3,4,5,6,7,8,9]).reshape(10,1)
        a = -1 * np.log(a)
        return   a  * mat.mult(np.transpose(y) , theKey)
    
    @staticmethod      
    def delta(z , a, y):
      return (a-y)




class QuadraticCost():

    @staticmethod
    def fn(a, y):
 
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):

        return (a-y) * sigmoid.prime(z)


class CrossEntropyCost():

    @staticmethod
    def fn(a, y):

        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z , a, y):

        return (a-y)
    


class ConvPoolLayer():

  ## assumption that the kernel is less than the input in size
   def __init__(self, PrevlayerSize, KernelSize, activationFunctions, PoolingBool=0, PoolingMapsFuncs=None, PoolingSize=(1,1,1)): 
          ''' PrevlayerSize is the size of the previous layer (3D Tensor),
              KernelSize is a tuple of the size of the Kernel (including the number of Channels it convolves with)
              activationFunctions is a list of activation for the convolutional
              if the layer contains Pooling or not and if it does
              PoolingMaps is a list of the pooling function for each layer (function has to return a scalar such as np.max() ) 
              PoolingSize is a tuple of the 2D pool receptive field  (eg: (2,2))
              self.PoolingLayerSize is size of the pooling layer (the map and the number of layers)
               '''

    if KernelSize[-1] != PrevlayerSize[-1]:
      raise Exception("Kernel is not the right size")    

    self.PrevlayerSize = PrevlayerSize
    self.ConvolutionalLayerSize = ( int( PrevlayerSize[0] - Kernels[0] + 1) , int( PrevlayerSize[1] - Kernels[1] + 1) , 
                                     len(PoolingMapsFuncs) )   
    self.PoolingBool = PoolingBool
    self.PoolingMapsFuncs = PoolingMapsFuncs
    self.PoolingSize = PoolingSize
    c = self.ConvolutionalLayerSize
    self.PoolingLayerSize = ( int( c[0] / PoolingSize[0]) , int( c[1] / PoolingSize[1] ) , 
                                     len(PoolingMapsFuncs) ) 
    
    self.activationFuncs = activationFunctions

    ## Will store the weighted input for the last forward pass call
    self.weightedInputs = None
    self.activation =None
    self.activations = None
    self.errors = None
    self.delAdelA = None 

    ### Init some Kernels
    self.kernelSize = KernelSize + (len(PoolingMapsFuncs),) 
    p = self.kernelSize
    ##tuple of (bias, weights) 
    self.Kernels = (np.random.randn(len(PoolingMapsFuncs)) * 0.001 , np.random.randn(p[0] , p[1] , p[2] , 
                                                                                p[3] ) * 0.001 )
    


    def poolingMap(self, batchActivations):
        poolingMaps = self.PoolingMapsFuncs
        listOfMapsByPool = [] ## In this case a map is a 4D tensor of a kernel map for the given batch
        # sizeOfPrev = batchActivations.shape[0:-1]
        listOfdelAdelAByPool = []


        ## iterating over each kernel
        
        for pool in poolingMaps:
          poolIndex = poolingMaps.index(pool)
          listOfMapsByImage = []
          listOfdelAdelAByImage = []
          poolingFunction = pool
          poolX = self.PoolingSize[0]
          poolY = self.PoolingSize[1]   ### 
          ###this is potentially 

          for inputCount in range(batchActivations.shape[-1]):
              input = batchActivations[:, :, poolIndex - 1 , inputCount]
              map  = np.zeros(poolX, poolY, 1,1)
              delAdelA = None

              ## This is with assumption that no overlapping is done in pooling
              listOfdelAdelAByNeuronX = []
              for x in range(self.PoolingLayerSize[0]):
                listOfdelAdelAByNeuronY = []

                for y in range(self.PoolingLayerSize[1]):
                  map[x][y][0][0]  =  poolingFunction(input[2 * x : 2 * x + poolX,
                                                            2 * y : 2 * y + poolY])
                  
                  ### Lets make the pooling function just maxpooling for now , L2/L1 average is quite ez
                  delAdelA = (input[2 * x : 2 * x + poolX, 2 * y : 2 * y + poolY] == map[x][y][0][0])[:,:]
                  listOfdelAdelAByNeuronY.append(delAdelA)
                listOfdelAdelAByNeuronX.append(np.concat(listOfdelAdelAByNeuronY, axis= 1))

                  
              listOfMapsByImage.append(map)
              listOfdelAdelAByImage.append(np.concat(listOfdelAdelAByNeuronX, axis= 0)[:,:,np.newaxis, np.newaxis])


        ### This may fail
        listOfMapsByPool.append(np.concat(listOfMapsByImage, axis= 3))
        self.delAdelA = np.concat(listOfdelAdelAByPool.append(np.concat(listOfdelAdelAByImage, axis= 3))  , axis=2)
        
        return np.concat(listOfMapsByPool, axis =2) 


    def feedForward(self, inputActivation): 

      ## input Activation is guranteed to be 4D tensor; for eg (5,8,3,10)  where 10 is the m
      kw = self.Kernels[1]   ## the 4D Tensor, weights  (5,8,3,2)
      kw = kw[:,:,:,np.newaxis,:]  ##(5,8,3,1,2)
      kb = self.Kernels[0]   ## the 1D vector, biases  (2,)
      ks = (self.KernelSize + (1,))     ## the size tuple of self.Kernels
      ks[-1] = ks[-2]
      ks[-2] = 1
      # sizeOfPrev = inputActivation.shape[0:-1] 


      listOfXWeightedInput = []

      for x in range(self.ConvolutionalLayerSize[0]):

          listOfYWeightedInput = []

          for y in range(self.ConvolutionalLayerSize[1]):
            qarray = inputActivation[ x : x+ks[0] , y:y+ks[1] , :,:,np.newaxis] ### (5,8,3,10,1)

            ### Convolution with double broadcasting
            doubleBroadcastArray = qarray * kw
            for number in range(0,3):
              doubleBroadcastArray = np.sum(doubleBroadcastArray, axis=0)  ### we get two dimensional, (10,2)

            xyneuronWeightedInput = np.flip(np.rot90(doubleBroadcastArray , k = 1, axes=(0,1)), axis=0)
            xyneuronWeightedInput = xyneuronWeightedInput[np.newaxis, np.newaxis,:,;]  ## 4D Tensor
            xyActivation = self.activationFunc.func(xyneuronWeightedInput)

            listOfYWeightedInput.append(xyneuronWeightedInput) 
            


          listOfXWeightedInput.append(np.concat(listOfYWeightedInput, axis= 1))


        layerWeightedInput = np.concat(listOfXWeightedInput, axis= 0)
        self.weightedInputs = layerWeightedInput.copy()


        for index in range(layerWeightedInput.shape[-2]):
          layerWeightedInput[:,:,index,:] =  self.activationFunctions[index].func(layerWeightedInput[:,:,index,:])

        self.activation = layerWeightedInput
        layerWeightedInput = None   


      if self.PoolingBool == 1:
        self.activations = self.poolingMap(self.activation)
      else:
        self.activations = self.activation


      def backprop(self, eta, batchSize, batchLabels, inputActivation, deeperDeltas, deeperWeights):
        ### BP1 does not apply to Conv Pool Layer 

      c = self.ConvolutionalLayerSize

      ### BP 2    
      if self.PoolingBool == 1 :

          deeperWeights = (deeperWeights.transpose()).reshape((self.PoolingLayerSize[0], 
                                                self.PoolingLayerSize[1], 
                                                self.PoolingLayerSize[2], deeperWeights.shape[0]), order='F')
          
        
          deeperWeights = deeperWeights[:,:,:,:,np.newaxis]  ### (12,12,3,10,1)
          deeperDeltas = deeperDeltas[np.newaxis,np.newaxis,np.newaxis]  ### (1,1,1,10,m)
          doubleBroadcastArray = deeperWeights * deeperDeltas  ## (12,12,3,10,m)
          deltasTimesWeights = np.sum(doubleBroadcastArray , axis=3)  ## (12,12,3,m) so there are m of sum(delta*weight) for each neuron in the layer
          s = deltasTimesWeights.shape      
          deltasTimesWeightsAugmented = np.kron(deltasTimesWeights , np.ones((self.poolingSize[0] ,
                                                  self.poolingSize[1]) + (1,1)))
          
          self.errors = np.zeros( ( c[0] , 
                                  c[1] ) + 
                                      ( c[2]  , batchSize ) )

          for index in range(c[2]):
            self.errors[:,:,index,:] = 
              self.delAdelA[:,:,index,:] *  self.activationFunctions[index].prime(self.weightedInputs[:,:,index,:]) * 
              deltasTimesWeightsAugmented[:,:,index,:]

          
        
      else:
        ## There is no pooling or downsampling that happens on the convolutional layer

        ## If The deeper layer is Fully Connected
        if len(deeperDeltas.shape) == 2:
          deeperWeights = (deeperWeights.transpose()).reshape((c[0], 
                                                c[1], 
                                                c[2], deeperWeights.shape[0]), order='F')
          
        
          deeperWeights = deeperWeights[:,:,:,:,np.newaxis]  ### (24,24,3,10,1)
          deeperDeltas = deeperDeltas[np.newaxis,np.newaxis,np.newaxis]  ### (1,1,1,10,m)
          doubleBroadcastArray = deeperWeights * deeperDeltas  ## (24,24,3,10,m)
          deltasTimesWeights = np.sum(doubleBroadcastArray , axis=3)  ## (12,12,3,m) so there are m of sum(delta*weight) for each neuron in the layer     
          self.errors = np.zeros( ( c[0] , 
                                  c[1] ) + 
                                      ( c[2]  , batchSize ) )

          for index in range(c[2]):
            self.errors[:,:,index,:] = 
              self.activationFunctions[index].prime(self.weightedInputs[:,:,index,:]) * 
                            deltasTimesWeights[:,:,index,:]
        
        
        ## The deeper layer is Convolutional Layer
        else:
          pass
          
      
      ## BP3 ; updating the Biases
      self.Kernels[0] = self.Kernels[0] - ( (eta) / batchSize ) * np.sum(np.sum(self.errors , axis=-1), axis=(0,1))


      iaR = inputActivation[:,:,:,np.newaxis,:]
      iaRslide = ( np.lib.stride_tricks.sliding_window_view(  iaR,
                                    (c[0], c[1]) + iaR.shape[2:] ))
      e = self.errors[np.newaxis, np.newaxis,:,:,np.newaxis,:,:]

      ## BP4 : updating the weights
      self.Kernels[1] = self.Kernels[1] - ( (eta) / batchSize ) *  ((iaRslide * e ).sum(axis=(2,3,-1)))   

    


class FullyConnected ():


    def __init__(self, PrevlayerSize, size, activationFunction=ReLu, costFunction=None):  
      ''' PrevlayerSize is a tuple of the previous layer, 
          size of the layer    '''


        self.PrevlayerSize = 1
        for k in PrevlayerSize[:-1]:
          self.PrevlayerSize *= k


        self.weights =  np.random.randn(size, PrevlayerSize)  * 0.0001
        self.biases =   np.random.randn(size,)  * 0.0001
        self.activationFunc = activationFunction


        ###Forbackprop
        self.activations = None
        self.weightedInputs = None
        self.errors = None
        self.costFunction = costFunction

        
    def feedForward(self, inputActivation): 


       ## This works for Convolutional or Fully Connected layer 
      inputActivation = inputActivation.reshape((self.PrevlayerSize,inputActivation.shapes[-1]), order = 'F')
      z = np.matmul(self.weights, inputActivation) + self.biases
      self.weightedInputs = z
      self.activations = (self.activationFunc.func(z))
      


   def backprop(self, eta, batchSize, batchLabels, inputActivation, deeperDeltas, deeperWeights):
      inputActivation = inputActivation.reshape((self.PrevlayerSize,inputActivation.shapes[-1]), order = 'F')
      NablaW = None
      Deltas = None
      NablaB = Deltas 



      #### BP1 / BP2 ######## 
      ## BP2
      if self.CostFunction == None:
          
        Deltas =   (np.matmult(numpy.transpose(deeperWeights) , deeperDeltas))   *  self.activationFunc.prime(self.weightedInputs)
          

      else :
        ##BP1  : this is for all cost and all activations incl. softmax

        Deltas = self.CostFunction.delta( self.weightedInputs , self.activation , batchLabels)
      
    ##BP3/4    
        NablaW = np.rot90(inputActivation[:,:,np.newaxis], k = 1, axes=(0,2))  * np.repeat( Deltas[:,:,np.newaxis], self.PrevlayerSize , axis =2)
        NablaB = Deltas
        
        
        self.weights = self.weights -   ( eta * np.sum(NablaW, axis=2)  / batchSize )
        self.biases = self.biases -   ( eta * np.sum(NablaB, axis=1)   / batchSize )
        
        
    
class inputLayer(object):

  def __init__(self, size):
    self.size = size
    self.activations = None
  



class Network(object):

    ## the first layer is an input layer

    def __init__(self, layers, etas, bacthSize=10): 
        ''' etas is a list of etas for each layer
            layers is a list of layers with the first being a tuple (i.e. the dimension of the input)
        
        '''



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
        nq = self.numlayers ## nq  layers w/ the input
        

        ## Feedforward to find all the activation
        batchImages = np.concat([x[:,:,:,np.newaxis] for x, y in batch], axis =-1) ## 4Dtensor 
        batchLabels = np.hstack([y for x, y in batch])
        self.layers[0].activations = batchImages

        ### for each layer, we feed forward
        for layerIndex in range(1,nq): 
          layer = self.layers[layerIndex]
          layer.feedForward(self.layers[layerIndex - 1].activations)


        ##########  ***** Back Propogatioon *******######### 
        ## The last layer
        #  eta, batchSize, batchLabels, inputActivation, deeperDeltas, deeperWeights 

        self.layers[-1].backprop(etas[-1], self.m, batchLabels=batchLabels,  
                                  self.layers[-2].activations, deeperDeltas=None, deeperWeights=None)
        
        
        ##Every other layer
        for layerIndex in range(2, nq): 
          self.layers[layerIndex * -1].backprop(etas[layerIndex * -1], self.m, batchLabels=None,  
                                  self.layers[(layerIndex * -1) - 1].activations,
                                    deeperDeltas = self.layers[(layerIndex * -1) + 1].errors , 
                                    deeperWeights = self.layers[(layerIndex * -1) + 1].weights)
          ### This code will break due to kernels in ConvLayers


        
    def SGD(self, batchSize, eta, epochs,
            trainingSet):  ## the epoch could later be made into a float
        leng = len(trainingSet)

        if leng == 0:
            return
        self.eta = eta
        self.m = batchSize
        m = self.m

        for T in range(0, epochs):
            ## shuffle up the training example and divy them up into batches of the size batchSize
            random.shuffle(trainingSet)
            setOfBatches = [trainingSet[q:q + m] for q in range(0, leng, m)]

            for Batch in setOfBatches:

              m = math.min(int(math.floor(m * 1.001)) , 500) 
              self.backprop(batch)




    ## returns the ANN's answer to the input image x
    def feedForward(self, batch):
      nq = self.numlayers - 1
     
      batchImages = np.concat([x[:,:,:,np.newaxis] for x, y in batch], axis =-1)
      

      batchLabels = [y for x, y in batch]
      batchActivations = [batchImages]


      for layerIndex in range(nq): 
        layer = self.layers[layerIndex + 1] 
        batchActivations.append(layer.feedForward(batchActivations[layerIndex]))

      return batchActivations[-1], batchLabels

    def eval(self, testImages, batchSize): ## measure accuracy
      accuracy = 0
      for interval in range(0, len(testImages), batchSize):
        x,y = self.feedForward(testImages[k:k+batchSize])
        answers =  [np.argmax(x[:,q])  for q in range(batchSize):]
        answers = zip(answers, y)
        accuracy += sum(int(x == y) for (x, y) in answers)
      return accuracy


