import tensorflow as tf

height = 28
width = 28
channels = 1
n_inputs = height * width


conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 10

with tf.name_scope("inputs"):
    X= tf.placeholder(tf.float32, shape=[None,n_inputs],name="X")
    X_reshaped = tf.reshape(X,shape=[-1,height,width,channels])
    y = tf.placeholder(tf.int32,shape=[None],name="y")
	

conv1 = tf.layers.conv2d(X_reshaped,filters=conv1_fmaps,kernel_size=conv1_ksize,
                            strides=conv1_stride,padding=conv1_pad,
                            activation=tf.nn.relu,name="conv1")


with tf.name_scope("inception"):
	with tf.name_scope("comp_5x5"):
		conv1x_1 = tf.layers.conv2d(conv1,filters=15,kernel_size=1,
								strides=1,padding=conv2_pad,
								activation=tf.nn.relu,name="1x1_1")
		conv5x   = tf.layers.conv2d(conv1x_1,filters= 7,kernel_size=5,
								strides=1,padding=conv2_pad,
								activation=tf.nn.relu,name="5x5")
	
	with tf.name_scope("comp_3x3"):
		conv1x_2 =tf.layers.conv2d(conv1,filters=15,kernel_size=1,
								strides=1,padding=conv2_pad,
								activation=tf.nn.relu,name="1x1_2")
							
		conv3x  = tf.layers.conv2d(conv1x_2,filters=7,kernel_size=3,
								strides=1,padding=conv2_pad,
								activation=tf.nn.relu,name="3x3")
	with tf.name_scope("comp_maxpool"):
		max_pool = pool3 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,1,1,1],padding="SAME")
	
		conv1x_3 = tf.layers.conv2d(max_pool,filters=15,kernel_size=1,
								strides=1,padding=conv2_pad,
								activation=tf.nn.relu,name="1x1_3")
	with tf.name_scope("comp_1x1"):
		conv1x_4 = tf.layers.conv2d(conv1,filters=15,kernel_size=1,
								strides=1,padding=conv2_pad,
								activation=tf.nn.relu,name="1x1_4")
	print(conv1x_4.get_shape())
	print(conv3x.get_shape())
	print(conv5x.get_shape())
	print(max_pool.get_shape())
	out = tf.concat([conv1x_4,conv3x,conv5x,max_pool],axis=3)

conv4 = tf.layers.conv2d(out,filters=32,kernel_size=3,
                            strides=2,padding=conv2_pad,
                            activation=tf.nn.relu,name="3x3_2")


with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

conv_fc1 = tf.layers.conv2d(pool3,filters=64,kernel_size=7,
							activation=tf.nn.relu,name="fc4")

conv_fc2 = tf.layers.conv2d(conv_fc1,filters=10,kernel_size=1,
							activation=None,name="fc5")
print(conv4.get_shape())
logits = tf.reshape(conv_fc2,shape=[-1,10])							

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


n_epochs = 1
batch_size = 20


from tensorflow.examples.tutorials.mnist import input_data
#change
mnist = input_data.read_data_sets("C:\\Users\\MTechLap\\Downloads\\mnist-train")

from datetime import datetime
now  = datetime.utcnow().strftime(("%Y%m%d%H%M%S"))
root_logdir = "tf_logs"
logdir = "{}/run-{}".format(root_logdir,now)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

init = tf.global_variables_initializer()
with tf.Session(config = config) as sess:
    file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch,y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op,feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        #acc_test  = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train)#, "Test accuracy:", acc_test)
        print("epoch %d" %(epoch))
        save_path = saver.save(sess, "./my_mnist_model")
   #print("moving to test")
   #print(accuracy.eval(feed_dict={X:mnist.train.images, y:mnist.train.labels}))
