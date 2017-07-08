import tensorflow as tf
import numpy as np
import re


def act_summary(x):
	 tensor_name = re.sub('%s_[0-9]*/' % "tower", '', x.op.name)
	 tf.summary.histogram(tensor_name+"/activations",x)
	 tf.summary.scalar(tensor_name+"/sparsity",tf.nn.zero_fraction(x))

def loss_summary(total_loss):
		loss_averages = tf.train.ExponentialMovingAverage(0.9,name="Avg")
		loss_averages_op = loss_averages.apply([total_loss])
		tf.summary.scalar(total_loss.op.name + '(raw)',total_loss)
		tf.summary.scalar(total_loss.op.name,loss_averages.average(total_loss))
		return  loss_averages_op
load_data =  tf.contrib.keras.datasets.cifar10.load_data
(X_train,y_train),(X_test,y_test) = load_data()
x_float = tf.cast(X_train,tf.float32)
x_float = tf.map_fn(lambda img: tf.image.per_image_standardization(img),x_float)


min_fraction_of_examples_in_queue = 0.4
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *min_fraction_of_examples_in_queue)

batch_size = 128
with tf.name_scope("inputs"):
	
	image_batch,label_batch = tf.train.shuffle_batch(
		[x_float,y_train],
		batch_size=batch_size,
		num_threads=2,
		capacity=10000,                         #min_queue_examples+3*batch_size,
		enqueue_many=True,
		min_after_dequeue=1000)#min_queue_examples)

	"""
	X= tf.placeholder(tf.float32, shape=[None,n_inputs],name="X")
	X_reshaped = tf.reshape(X,shape=[-1,height,width,channels])
	y = tf.placeholder(tf.int32,shape=[None],name="y")
	"""
	X_reshaped = image_batch
	y = label_batch
	y =tf.cast(y,tf.int32)
	y = tf.reshape(y,shape=[-1])



conv1 = tf.layers.conv2d(X_reshaped,filters=64,kernel_size=5,strides=1,padding="SAME",name="conv1")
act_summary(conv1)
pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool1")
norm1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
conv2 = tf.layers.conv2d(norm1,filters=64,kernel_size=5,strides=1,padding="SAME",name="conv2")
act_summary(conv2)
norm2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool2")
s= pool2.get_shape()

h,w = s[1].value,s[2].value
out = tf.reshape(pool2,shape=[-1,64*h*w])
#print(out.get_shape())
fc3 = tf.layers.dense(out,384,activation=tf.nn.relu)
act_summary(fc3)
fc4 = tf.layers.dense(fc3,192,activation=tf.nn.relu)
act_summary(fc3)
logits = tf.layers.dense(fc4,10,activation=None)
act_summary(logits)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y)
loss = tf.reduce_mean(xentropy)

NUM_EPOCHS_PER_DECAY = 350.0
MOVING_AVERAGE_DECAY = .9999
DECAY_FACTOR = 0.1
num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/batch_size
decay_steps  = int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY)
global_step = tf.Variable(0,name="global_step",trainable=False)
lr = tf.train.exponential_decay(.1,
								global_step,
								decay_steps,
								DECAY_FACTOR,
								staircase=True)
tf.summary.scalar('learning_rate',lr)
loss_averages_op = loss_summary(loss)

with tf.name_scope("train"):
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.GradientDescentOptimizer(.1)
		grads = opt.compute_gradients(loss)
	apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name,var)
	for grad,var in grads:
		if grad is not None:
			tf.summary.histogram(var.op.name + "/gradients",grad)
	variable_averages = tf.train.ExponentialMovingAverage(
				MOVING_AVERAGE_DECAY,global_step)
	variable_averages_op = variable_averages.apply(tf.trainable_variables())
	with tf.control_dependencies([apply_gradient_op,variable_averages_op]):
		training_op = tf.no_op(name="train")
"""
with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.GradientDescentOptimizer(.1)
    training_op = optimizer.minimize(loss,global_step)
"""


tf.summary.image("images",X_reshaped)
import time
from datetime import datetime
class _LoggerHook(tf.train.SessionRunHook):
	"""Logs loss and runtime."""
	def begin(self):
		self._step = -1
		self._start_time = time.time()
		self.log_frequency = 10
	def before_run(self, run_context):
		self._step += 1
		return tf.train.SessionRunArgs(loss)  # Asks for loss value.

	def after_run(self, run_context, run_values):
		if self._step % self.log_frequency == 0:
			current_time = time.time()
			duration = current_time - self._start_time
			self._start_time = current_time

			loss_value = run_values.results
			examples_per_sec = self.log_frequency * batch_size / duration
			sec_per_batch = float(duration / self.log_frequency)

			format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
				'sec/batch)')
			print (format_str % (datetime.now(), self._step, loss_value,
				examples_per_sec, sec_per_batch))
		#print(run_values.results)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
global_step = tf.contrib.framework.get_or_create_global_step()
with tf.train.MonitoredTrainingSession(
	checkpoint_dir="./tf_logs2",
	hooks=[tf.train.StopAtStepHook(last_step=1000000),
		tf.train.NanTensorHook(loss),
		_LoggerHook()],
	config=config) as mon_sess:
	while not mon_sess.should_stop():
		mon_sess.run(training_op)
