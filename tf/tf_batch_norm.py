import tensorflow as tf
from tensorflow.contrib.layers import batch_norm,fully_connected

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

#matrix of mini-batch inputs
X = tf.placeholder(tf.float32, shape=(None,n_inputs), name="X")
y = tf.placeholder(tf.int64,shape=(None),name="y")
#tells batch norm whether to use emperical or population mean
is_training = tf.placeholder(tf.bool, shape=(),name="is_training")

bn_params = {
    'is_training': is_training,
    'decay': .99, #running average weight
    'updates_collections':None
}


repeat =False
if repeat:
    hidden1 = fully_connected(X,n_hidden1, scope = "hidden1",
                        normalizer_fn=batch_norm, normalizer_params=bn_params)
    hidden2 = fully_connected(hidden1,n_hidden2, scope="hidden2",
                        normalizer_fn=batch_norm, normalizer_params=bn_params)
    #output (affine-layer)
    logits = fully_connected(hidden2,n_outputs,activation_fn=None,scope="outputs",
                        normalizer_fn=batch_norm,normalizer_params=bn_params)
else:
    with tf.contrib.framework.arg_scope(
                [fully_connected],
                normalizer_fn = batch_norm,
                normalizer_params = bn_params):
            hidden1 = fully_connected(X,n_hidden1,scope="hidden1")
            hidden2 = fully_connected(hidden1,n_hidden2,scope="hidden2")
            logits = fully_connected(hidden2,n_outputs,scope="outputs",
                        activation_fn=None)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(\
            labels=y,logits=logits)
    loss = tf.reduce_mean(xentropy,name="loss")


learning_rate = 0.01

clip = False
if not clip:
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
else:
    #perform gradient clipping
    thresh = 1.0
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad,-thresh,thresh),var)
                        for grad,var in grads_and_vars]
        training_op =optimizer.apply_gradients(capped_gvs)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()

from tensorflow.examples.tutorials.mnist import input_data
#change
mnist = input_data.read_data_sets("/home/lie/Downloads/MNIST")

n_epochs = 400
batch_size = 50
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples// batch_size):
            X_batch,y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op,feed_dict={is_training:True,X:X_batch,y:y_batch})
        accuracy_score = accuracy.eval(feed_dict={is_training:False,X:mnist.test.images,y:mnist.test.labels})
        print(accuracy_score)
