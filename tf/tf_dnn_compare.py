import tensorflow as tf
from functools import partial
from mnist import load_data_wrapper
import numpy as np
from datetime import datetime
import random
import copy
"""
5-layer DNN for comparing digits

"""
n_inputs = 28 * 28
n_hidden = 100
n_outputs = 1
n_epochs = 10
learning_rate = 1
batch_size = 50



is_training = tf.placeholder(tf.bool,shape=(),name="is_training")
he_init = tf.contrib.layers.variance_scaling_initializer()
def batch_layer(X,bn=True,units=n_hidden,name="hidden"):
    global n_hidden

    batch_norm_momentum=0.99
    batch_norm_layer = partial(
                        tf.layers.batch_normalization,
                        training=is_training,
                        momentum=batch_norm_momentum
    )
    dense_layer     = partial(
                        tf.layers.dense,
                        kernel_initializer=he_init,
                        units = units
    )
    with tf.name_scope(name):
        dense = dense_layer(X)
        return batch_norm_layer(dense) if bn else dense

elu_tf = tf.nn.elu

Xa = tf.placeholder(tf.float32, shape=(None,n_inputs),name="Xa")
with tf.name_scope("dnna"):
    #network A
    dnna_h1 = elu_tf(batch_layer(Xa))
    dnna_h2 = elu_tf(batch_layer(dnna_h1))
    dnna_h3 = elu_tf(batch_layer(dnna_h2))
    dnna_h4 = elu_tf(batch_layer(dnna_h3))
    dnna_h5 = elu_tf(batch_layer(dnna_h4))
    logits = tf.layers.dense(dnna_h5,units=1)
Xb = tf.placeholder(tf.float32, shape=(None,n_inputs),name="Xb")
with tf.name_scope("dnnb"):
    #network B
    dnnb_h1 = elu_tf(batch_layer(Xb))
    dnnb_h2 = elu_tf(batch_layer(dnnb_h1))
    dnnb_h3 = elu_tf(batch_layer(dnnb_h2))
    dnnb_h4 = elu_tf(batch_layer(dnnb_h3))
    dnnb_h5 = elu_tf(batch_layer(dnnb_h4))
y = tf.placeholder(tf.float32,   shape=(None),name="y")

with tf.name_scope("output"):
    #single output unit
    dnn_out = tf.concat([dnna_h5,dnnb_h5],axis=1)
    logits = batch_layer(dnn_out,bn=False,units=n_outputs,name="logit")


with tf.name_scope("loss"):
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels =y, logits=logits)
    loss = tf.reduce_mean(xentropy,name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #account for batch norm updates: mean,gamma,beta
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.equal(tf.cast(tf.greater_equal(logits,0.5),tf.float32),y)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

def np_to_tf(training_data):
    """
    training data comes in as [(x,y),(x,y),...]
    outputs [[x],[x],[x],...],[[y],[y],[y],..]
    ravels training data and puts it in 1D array form that
    tensorflow optimizer likes (for simulaneous batch training)
    """
    unzip = list(zip(*training_data))
    x = list(unzip[0])
    y = list(unzip[1])
    x = [t.ravel() for t in x]
    y = [t.ravel() for t in y]
    return x,y

def batch_gen(training_data,batch_size):
    """
    splits training data into batches of tuples that contain images of the same digit
    and images of different digits
    """
    training_data =  copy.deepcopy(training_data)
    xa_batchs = []
    xb_batchs =[]
    y_batchs = []
    xa_batch =[]
    xb_batch = []
    y_batch = []
    start = 0
    num_batches = len(training_data)// batch_size
    batch_num =0
    end_early = False
    while batch_num != num_batches:
        i =0
        element = training_data.pop(start)
        #mini_batch.push(element)
        start = 0
        while i < batch_size:
            try:
                ex = training_data[start]
            except IndexError:
                end_early = True
                break
            same =  (i%2 == 0 and np.argmax(element[1])==np.argmax(ex[1]))
            different = (i%2 ==1 and np.argmax(element[1])!=np.argmax(ex[1]))
            if  same or different:
                xa_batch.append(element[0])
                xb_batch.append(training_data.pop(start)[0])
                y_batch.append(same)
                i+=1
                try:
                    element = training_data[start+1]
                except IndexError:
                    end_early = True
                    break
            start+=1
        if end_early:
            break

        batch_num+=1
        xa_batch = [x.ravel() for x in xa_batch]
        xb_batch = [x.ravel() for x in xb_batch]
        xa_batchs.append(xa_batch)
        xb_batchs.append(xb_batch)
        y_batchs.append(y_batch)
        xa_batch =[]
        xb_batch = []
        y_batch = []
    return zip(xa_batchs,xb_batchs,y_batchs)

train,valid,test = load_data_wrapper()

train = list(train)



init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    now  = datetime.utcnow().strftime(("%Y%m%d%H%M%S"))
    root_logdir = "tf_logs"
    logdir = "{}/run-{}".format(root_logdir,now)

    file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())
    cost_log = tf.summary.scalar("Cost",loss)
    num_batch = len(train) //batch_size
    for epoch in range(n_epochs):

        random.shuffle(train)
        batch_index = 0
        batchs = batch_gen(train,batch_size)
        print(len(train))
        for xa_batch,xb_batch,y_batch in batchs:
            sess.run(training_op,feed_dict={Xa:xa_batch,Xb:xb_batch,y:y_batch,is_training:True})
            if batch_index%10 ==0:
                step = epoch*num_batch +batch_index
                cost_now = cost_log.eval(feed_dict={Xa:xa_batch,Xb:xb_batch,y:y_batch,is_training:False})
                print(accuracy.eval(feed_dict={Xa:xa_batch,Xb:xb_batch,y:y_batch,is_training:False}))
                file_writer.add_summary(cost_now,step)
            batch_index+=1

        print("EPOCH: %d " %epoch,end="")
        print("\n")
