from NN.src import mnist_loader
import network2
import json

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.Network([784, 20, 10])
# net.SGD(training_data, 1, 7, 3.0, test_data=None)

e_c,e_a,t_c,t_a = net.SGD(training_data, 30, 7, 3.2,
            lmbda = 50.0,
            evaluation_data=validation_data,
            monitor_evaluation_cost=True,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)


data = {
 'ec': e_c , '2' : e_a, '3' : t_c, '4' : t_a
}

f = open('data', "w")
json.dump(e_c,e_a,t_c,t_a)
f.close()

print (net.eval(training_data))
print (net.eval(validation_data))

