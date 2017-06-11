import tensorflow as tf
from mnist import load_data_wrapper
from datetime import datetime
from functools import partial
import numpy as np
import random

"""

5-layer DNN using the layers module

"""


n_inputs = 28*28
n_outputs = 5

#hyper-params
n_hidden = 100
learning_rate =  2
n_epochs = 400
batch_size = 50
batch_norm_momentum=0.99

#sometimes it is at the end and sometimes not
bn_at_end = False
bn=True

if bn_at_end and not bn:
    raise Exception("Cant have no hidden layer bn but outer layer bn")

X = tf.placeholder(tf.float32, shape=(None,n_inputs),name="X")
y = tf.placeholder(tf.int64,   shape=(None),name="y")


with tf.name_scope("dnn"):
    he_init = tf.contrib.layers.variance_scaling_initializer()
    is_training = tf.placeholder(tf.bool,shape=(),name="is_training")

    def batch_layer(X,bn=True,name="hidden"):
        global n_hidden
        batch_norm_layer = partial(
                            tf.layers.batch_normalization,
                            training=is_training,
                            momentum=batch_norm_momentum
        )
        dense_layer     = partial(
                            tf.layers.dense,
                            kernel_initializer=he_init,
                            units = n_hidden
        )
        with tf.name_scope(name):
            dense = dense_layer(X)
            return batch_norm_layer(dense) if bn else dense



    elu_tf = tf.nn.elu

    #network
    h1 = elu_tf(batch_layer(X,bn))
    h2 = elu_tf(batch_layer(h1,bn))
    h3 = elu_tf(batch_layer(h2,bn))
    h4 = elu_tf(batch_layer(h3,bn))
    h5 = elu_tf(batch_layer(h4,bn))
    logits = batch_layer(h5,bn=bn_at_end,name="logits")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = y,logits =logits)
    loss = tf.reduce_mean(xentropy,name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #account for batch norm updates: mean,gamma,beta
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))



def numpy_to_tf(training_data):
    unzip = list(zip(*training_data))
    x = list(unzip[0])
    y = list(unzip[1])
    return x,y
def extract_first_n(tf_training_data,n):
    x,y = tf_training_data
    x = [t.ravel() for t,m in zip(x,y) if (sum(m[:n])!=0 if not isinstance(m,np.int64) else (m<n))]
    f = lambda t : np.argmax(t) if not isinstance(t,np.int64) else t
    y = [f(t) for t in y if (sum(t[:n])!=0  if not isinstance(t,np.int64) else (t<n))]
    return x,y



train,valid,test = load_data_wrapper()

init = tf.global_variables_initializer()
saver = tf.train.Saver()

test_data = True
valid_data = False
with tf.Session() as sess:
    init.run()

    now  = datetime.utcnow().strftime(("%Y%m%d%H%M%S"))
    root_logdir = "tf_logs"
    logdir = "{}/run-{}".format(root_logdir,now)

    file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())
    cost_log = tf.summary.scalar("Cost",loss)
    n = n_outputs
    #perform some conversion to make train list into X,y
    train = list(zip(*extract_first_n(numpy_to_tf(train),n)))

    num_train = len(train)

    valid = extract_first_n(numpy_to_tf(valid),n)
    test = extract_first_n(numpy_to_tf(test),n)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            ckpt_save = logdir+"/"+"model.ckpt"
            save_path = saver.save(sess,ckpt_save)

        #shuffle X,y together
        random.shuffle(train)
        X_train,y_train = zip(*train)

        mini_batchs = [
            (X_train[k:k+batch_size],y_train[k:k+batch_size]) for k in range(0,num_train,batch_size)
        ]
        num_batch = len(mini_batchs)
        for batch_index,batch in enumerate(mini_batchs):
            X_batch,y_batch = batch

            sess.run(training_op,feed_dict={X:X_batch,y:y_batch,is_training:True} if bn else {X:X_batch,y:y_batch})

            if batch_index%10 ==0:
                step = epoch*num_batch +batch_index
                cost_now = cost_log.eval(feed_dict={X:X_batch,y:y_batch,is_training:False} if bn else {X:X_batch,y:y_batch})
                file_writer.add_summary(cost_now,step)

        print("EPOCH: %d " %epoch,end="")
        if test_data:
            X_test,y_test =test
            accuracy_score = accuracy.eval(feed_dict=({X:X_test,y:y_test,is_training:False} if bn else {X:X_test,y:y_test}))
            print(", test: %f"  % accuracy_score,end="")
        if valid_data:
            X_valid,y_valid = valid
            accuracy_score = accuracy.eval(feed_dict=({X:X_valid,y:y_valid,is_training:False} if bn else {X:X_valid,y:y_valid}))
            print(", valid: %f" % accuracy_score,end="")
        print("\n")

    save_path = saver.save(sess,logidr+"/"+"model.ckpt")
