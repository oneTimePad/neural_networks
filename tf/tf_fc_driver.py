from tf_fcnetwork import FCNetwork,relu,sigmoid,quadratic_cost,softmax,cross_entropy
from mnist import load_data_wrapper


network_topology = [784,30,10]
act =[relu,softmax]

net = FCNetwork(network_topology,act,cross_entropy,.1)

train,valid,test = load_data_wrapper()
epochs = 10
mini_batch = 10
net.train(train,epochs,mini_batch,valid)
