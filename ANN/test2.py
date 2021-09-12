import mnist_loader
import ANN
# import network
import time

start_time = time.time()

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

##  batchSize, eta (float), epochs,
net = ANN.Network([784, 1])
net.SGD(7, 3.9, 1, training_data)
print( "Success Rate : " + str(net.eval(test_data)))


while True:
    bs = input()
    if bs == 0:
        break
    eta = input()
    epo = input()


    net.SGD(bs, eta, epo, training_data)
    print("Success Rate : " + str(net.eval(test_data)))


print("--- %s seconds ---" % (time.time() - start_time))


## GOOD HyperParameters
# net.SGD(6 or 7, 3.6 or 3.7, 1, training_data)
# net.SGD(9, 3.9, 1, training_data)  ~ 93.33%
# net.SGD(16, 3.6, 1, training_data)


#

''' Nielson's result
Epoch 0: 9129 / 10000
Epoch 1: 9295 / 10000
Epoch 2: 9348 / 10000
...
Epoch 27: 9528 / 10000
Epoch 28: 9542 / 10000
Epoch 29: 9534 / 10000
'''




'''
Success Rate : 9320
Success Rate : 9501
18
3.0
1
18
Success Rate : 9530
18
3.2
1
18
Success Rate : 9532
25
3.0
1
25
Success Rate : 9559
35
2.5
1
35
Success Rate : 9577
35
2.7
1
35
Success Rate : 9586
35
2.1
1
35
Success Rate : 9581
25
2.7
1
25
Success Rate : 9601
25
2.9
1
25
Success Rate : 9586'''
