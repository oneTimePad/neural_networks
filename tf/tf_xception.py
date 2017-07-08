import tensorflow as tf

height = 28
width = 28
channels = 1
n_inputs = height*width

height = 32
width = 32
channels =32
n_inputs = 32*32
import numpy as np
load_data =  tf.contrib.keras.datasets.cifar10.load_data
(X_train,y_train),(X_test,y_test) = load_data()
x_float = tf.cast(X_train,tf.float32)
x_float = tf.map_fn(lambda img: tf.image.per_image_standardization(img),x_float)


with tf.name_scope("inputs"):
	
	image_batch,label_batch = tf.train.shuffle_batch(
		[x_float,y_train],
		batch_size=128,
		num_threads=2,
		capacity=5000,
		enqueue_many=True,
		min_after_dequeue=1000)

	"""
	X= tf.placeholder(tf.float32, shape=[None,n_inputs],name="X")
	X_reshaped = tf.reshape(X,shape=[-1,height,width,channels])
	y = tf.placeholder(tf.int32,shape=[None],name="y")
	"""
	X_reshaped = image_batch
	y = label_batch
	y =tf.cast(y,tf.int32)
	y = tf.reshape(y,shape=[-1])
	print(y.get_shape())
	

conv_pading="SAME"
pool_pading="VALID"
with tf.name_scope("factored_inception"):
	with tf.variable_scope("branch1"):
		branch1 = tf.layers.conv2d(X_reshaped,filters=32,kernel_size=1,strides=1,padding=conv_pading,activation=tf.nn.relu,name="Conv1x_1")
		branch1 = tf.layers.conv2d(branch1,filters=32,kernel_size=[1,7],strides=1,padding=conv_pading,activation=tf.nn.relu,name="Conv1x7_1")
		branch1 = tf.layers.conv2d(branch1,filters=32,kernel_size=[7,1],strides=1,padding=conv_pading,activation=tf.nn.relu,name="Conv7x1_1")
		branch1 = tf.layers.conv2d(branch1,filters=32,kernel_size=[1,7],strides=1,padding=conv_pading,activation=tf.nn.relu,name="Conv1x7_2")
		branch1 = tf.layers.conv2d(branch1,filters=32,kernel_size=[7,1],strides=2,padding=conv_pading,activation=tf.nn.relu,name="Conv7x1_2")
		#print(branch1.get_shape())
	with tf.variable_scope("branch2"):
		branch2 = tf.layers.conv2d(X_reshaped,filters=32,kernel_size=1,strides=1,padding=conv_pading,activation=tf.nn.relu,name="Conv1x_1")
		branch2 = tf.layers.conv2d(branch2,filters=32,kernel_size=[7,1],strides=1,padding=conv_pading,activation=tf.nn.relu,name="Conv1x7_1")
		branch2 = tf.layers.conv2d(branch2,filters=32,kernel_size=[1,7],strides=2,padding=conv_pading,activation=tf.nn.relu,name="Conv7x1_1")
		#print(branch2.get_shape())
	with tf.variable_scope("branch3"):
		branch3 = tf.layers.conv2d(X_reshaped,filters=32,kernel_size=1,strides=1,padding=conv_pading,activation=tf.nn.relu,name="Conv1x_1")
		branch3 = tf.nn.max_pool(branch3,ksize=[1,2,2,1],strides=[1,2,2,1],padding=pool_pading,name="Max_pool_1")
		#print(branch3.get_shape())
	with tf.variable_scope("branch4"):
		branch4 = tf.layers.conv2d(X_reshaped,filters=32,kernel_size=1,strides=2,padding=conv_pading,activation=tf.nn.relu,name="Conv1x_1")
		#print(branch4.get_shape())

	out = tf.concat([branch1,branch2,branch3,branch4],axis=3)
	print(out.get_shape())

conv = tf.layers.conv2d(out,filters=32,kernel_size=1,strides=2,activation=tf.nn.relu,padding=conv_pading,name="Conv1x")
print("!")
print(conv.get_shape())
print("!")
with tf.name_scope("depthwise_sep"):
	#depth = tf.nn.depthwise_conv2d(X_reshaped,filter=[2.0,2.0,1.0,1.0],strides=[1,1,1,1],padding=conv_pading)
	#depth = tf.nn.separable_conv2d(X_reshaped,depthwise_filter=[3,3,32,1],strides=[1,1,1,1],pointwise_filter=[1,1,32,64],padding=conv_pading,name="Conv_sep")
	depth = tf.contrib.layers.separable_conv2d(conv,32,depth_multiplier=1,kernel_size=[3,3],padding=conv_pading)
print(depth.get_shape())
with tf.name_scope("fc"):
	conv_fc3 = tf.layers.conv2d(depth,filters=64,activation=tf.nn.relu,kernel_size=8,strides=1,padding="VALID",name="fc3")
	conv_fc4 = tf.layers.conv2d(conv_fc3,activation=None,filters=10,kernel_size=1,strides=1,padding="VALID",name="fc4")
	logits = tf.reshape(conv_fc4,shape=[-1,10])
print(logits.get_shape())
with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.GradientDescentOptimizer(.1)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


n_epochs = 1
batch_size = 50
import time
class _LoggerHook(tf.train.SessionRunHook):
	"""Logs loss and runtime."""
	def begin(self):
		self._step = -1
		self._start_time = time.time()

	def before_run(self, run_context):
		self._step += 1
		return tf.train.SessionRunArgs(loss)  # Asks for loss value.

	def after_run(self, run_context, run_values):
		"""
		if self._step % FLAGS.log_frequency == 0:
			current_time = time.time()
			duration = current_time - self._start_time
			self._start_time = current_time

			loss_value = run_values.results
			#examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
			#sec_per_batch = float(duration / FLAGS.log_frequency)

			format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
				'sec/batch)')
			print (format_str % (datetime.now(), self._step, loss_value,
				examples_per_sec, sec_per_batch))
		"""
		print(run_values.results)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
global_step = tf.contrib.framework.get_or_create_global_step()
with tf.train.MonitoredTrainingSession(
	checkpoint_dir="./tf_logs",
	hooks=[tf.train.StopAtStepHook(last_step=3),
		tf.train.NanTensorHook(loss),
		_LoggerHook()],
	config=config) as mon_sess:
	while not mon_sess.should_stop():
		mon_sess.run(training_op)

"""
from tensorflow.examples.tutorials.mnist import input_data
#change
mnist = input_data.read_data_sets("C:\\Users\\MTechLap\\Downloads\\mnist-train")
"""
from datetime import datetime
now  = datetime.utcnow().strftime(("%Y%m%d%H%M%S"))
root_logdir = "tf_logs"
logdir = "{}/run-{}".format(root_logdir,now)





"""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import numpy as np
init = tf.global_variables_initializer()
with tf.Session(config = config) as sess:
	file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())
	print("MOVING TO TRAINING")
	init.run()
	
	with tf.train.MonitoredTra
	
	
	for epoch in range(n_epochs):
		for iteration in range(mnist.train.num_examples // batch_size):
			X_batch,y_batch = mnist.train.next_batch(batch_size)
			print(y_batch)
			sess.run(training_op,feed_dict={X: X_batch, y: y_batch})
		acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
		#acc_test  = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
		print(epoch, "Train accuracy:", acc_train)#, "Test accuracy:", acc_test)
		print("epoch %d" %(epoch))
		save_path = saver.save(sess, "./my_mnist_model")
"""

   
   #print("moving to test")
   #print(accuracy.eval(feed_dict={X:mnist.train.images, y:mnist.train.labels}))




