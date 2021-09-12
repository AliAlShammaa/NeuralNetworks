
import ANN2
import time
import math

start_time = time.time()
training_data = [(0,0),(math.pi / 6, 1 / 2) , (math.pi / 4, 1/(2**(1/2))) , (math.pi / 3, (3**(1/2)) / 2  ) ,
                 (math.pi / 2, 1) , (math.pi, 0)]
test_data = [(3,math.sin(3)),(0,0)]


##  batchSize, eta (float), epochs,
net = ANN2.Network([1, 10 , 1])
net.SGD(1, 0.1, 1, training_data)
print( "Success Rate : " + str(net.eval(test_data)))


print("--- %s seconds ---" % (time.time() - start_time))

