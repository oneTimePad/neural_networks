
import tensorflow as tf




xception = tf.contrib.keras.applications.Xception(
											include_top=False,
											weights='imagenet',
											pooling='avg')



logits = tf.contrib.keras.layers.Dense(5,input_shape(2048,))(xception)
print(logits)
import sys;sys.exit(1)

import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
from datetime import datetime

"""
Inception transfer learning for flower data set based on
https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10
and
https://github.com/ageron/handson-ml/blob/master/13_convolutional_neural_networks.ipynb
"""

MODE = 'train'
HEIGHT= 299
WIDTH = 299
DEPTH = 3
BATCH_SIZE = 50
NUM_EPOCHS = 50
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500
FRACTION_IN_QUEUE = 0.4
NUM_EPOCHS_PER_DECAY=15
NUM_OUT = 5
INITIAL_LEARNING_RATE = 0.01
DECAY_RATE = 0.96
LOG_FREQUENCY = 10
LOGDIR = '/tmp/xception_new'
FLOWERS_DIR = 'C:/Users/MTechLap/Desktop/models/tutorials/image/cifar10/flowers_train.bin'
#INCEPTION_V3_CHECKPOINT = 'C:/Users/MTechLap/Desktop/models/tutorials/image/cifar10/datasets/inception/inception_v3.ckpt'
def parse_data(filenames):
	"""
	args:
		filesnames (input)
	returns:
		key:= file key from reader
		image:= parsed images from reader
		label:= parsed label from reader
	"""

	filenames_queue = tf.train.string_input_producer(filenames)
	label_bytes = 1
	image_bytes = WIDTH*HEIGHT*DEPTH
	#format for flower images is like cifar10
	#(label_byte)+linear_image
	record_bytes = label_bytes + image_bytes
	#read record bytes each for each sample
	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	#unique entry identifier, value(string)
	key,value = reader.read(filenames_queue)
	#convert to bytes
	record_bytes = tf.decode_raw(value,tf.uint8)
	#extract label
	label = tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)

	image_part_linear = tf.strided_slice(record_bytes,[label_bytes],[label_bytes+image_bytes])
	#image = tf.reshape(image_part_linear,[DEPTH,HEIGHT,WIDTH])
	#image = tf.transpose(image,[1,2,0])
	image = tf.reshape(image_part_linear,[HEIGHT,WIDTH,DEPTH])
	return (key,image,label)

def format_input(image,label):
	image = tf.cast(image,tf.float32)
	#add some randomness
	#flip = tf.image.random_flip_left_right(image)
	#bright_change = tf.image.random_brightness(flip,max_delta=63)
	#contrast = tf.image.random_contrast(bright_change,lower=0.2,upper=1.8)
	#hue = tf.image.random_hue(contrast,0.2)
	#inception expect inputs [-1.0,1.0]
	#bright_change = image
	scaled = tf.multiply(image,2.0/255.0)

	scaled = tf.subtract(scaled,1.0)
	#make batch functions happy (static shape)
	scaled.set_shape([HEIGHT,WIDTH,DEPTH])
	label.set_shape([1])
	return(scaled,label)

def gen_batch(image,label):
	min_queue_size = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN*
						FRACTION_IN_QUEUE)
	num_threads = 4
	#generate random training batches
	image_batch,label_batch = tf.train.shuffle_batch(
		[image,label],
		batch_size=BATCH_SIZE,
		num_threads = num_threads,
		capacity =min_queue_size +3*BATCH_SIZE,
		min_after_dequeue = min_queue_size
	)
	tf.summary.image('images',image_batch)
	#reshape to labels to vector
	return image_batch,tf.reshape(label_batch,[BATCH_SIZE])

#preprocessing and training
with tf.device('/cpu:0'):
	print('starting to load data. please wait...')
	key,image,label  = parse_data([FLOWERS_DIR])
	scaled,label = format_input(image,label)
	image_batch,label_batch = gen_batch(scaled,label)

training = tf.placeholder_with_default(False, shape=[])
"""
with slim.arg_scope(inception.inception_v3_arg_scope()):
	old_logits,end_points = inception.inception_v3(
					image_batch,num_classes=1001,is_training=False)

	#saver = tf.train.Saver()
	#reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	#reuse_vars_dict = {var.name: var for var in reuse_vars}
	#original_saver = tf.train.Saver(reuse_vars_dict)
	original_saver = tf.train.Saver()
"""
with tf.name_scope('flower_output'):
	pre_logits = end_points['PreLogits']
	print(pre_logits.op.name)
	#logits = tf.layers.conv2d(pre_logits,kernel_size=1,strides=1,filters=5,padding='SAME',name='flower_logits')
	#prediction = tf.nn.softmax(logits,name='flower_softmax')
	#logits_full = tf.reshape(logits,shape=[-1,NUM_OUT])
	#logits_full = tf.squeeze(logits)
	pool = tf.squeeze(pre_logits)
	logits_full = tf.layers.dense(pool,5,kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name='flower_logits')
with tf.name_scope('flower_loss'):
	label64 = tf.cast(label_batch,tf.int64)
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
				labels=label64,logits =logits_full)
	loss = tf.reduce_mean(xentropy,name='flower_loss')
	#get moving average
	loss_avg = tf.train.ExponentialMovingAverage(0.9,name='flower_avg')
	#get the moving average ops (create shadow variables)
	loss_avg_op = loss_avg.apply([loss])
	#log loss and shadow variables for avg loss
	tf.summary.scalar(loss.op.name+' (raw)',loss)
	tf.summary.scalar(loss.op.name,loss_avg.average(loss))

with tf.name_scope('flower_train'):
	global_step = tf.contrib.framework.get_or_create_global_step()#tf.Variable(0,trainable=False)
	num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/BATCH_SIZE
	decay_steps = num_batches_per_epoch*NUM_EPOCHS_PER_DECAY
	#decay learning rate in discrete fashion
	lr = tf.train.exponential_decay(
		INITIAL_LEARNING_RATE,
		global_step,
		decay_steps,
		DECAY_RATE,
		staircase=True
	)
	tf.summary.scalar('learning_rate',lr)

	with tf.control_dependencies([loss_avg_op]):
		train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='flower_logits')
		print(train_vars)
		sgd_op = tf.train.GradientDescentOptimizer(lr)
		grads = sgd_op.compute_gradients(loss,var_list=train_vars)
	apply_grads_op = sgd_op.apply_gradients(grads,global_step=global_step)
	#track values of gradients and variables
	for grad,var in grads:
		if grad is not None:
			tf.summary.histogram(var.op.name+'/gradient',grad)
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name,var)
	#get a moving average of all variables
	variable_avg = tf.train.ExponentialMovingAverage(
			0.9,global_step)
	variable_avg_op = variable_avg.apply(tf.trainable_variables())

	with tf.control_dependencies([apply_grads_op,variable_avg_op]):
		train_op = tf.no_op(name='flower_train_op')

with tf.name_scope('eval'):
	correct = tf.nn.in_top_k(logits_full,label_batch,1)
	accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

a = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='flower_logits')]
class _LoggerHook(tf.train.SessionRunHook):

	def begin(self):
		self._step = -1
		self._start_time = time.time()
	def before_run(self,run_context):
		self._step+=1
		#print(run_context.original_args.fetches)
		return tf.train.SessionRunArgs([loss,accuracy])

	def after_run(self,run_context,run_values):
		#print(run_values.results[2])
		if self._step % LOG_FREQUENCY ==0:
			current_time = time.time()
			duration = current_time - self._start_time
			self._start_time = current_time

			loss_value = run_values.results[0]
			acc = run_values.results[1]
			
			examples_per_sec = LOG_FREQUENCY/duration
			sec_per_batch = duration / LOG_FREQUENCY

			format_str = ('%s: step %d, loss = %.2f, acc = %.2f (%.1f examples/sec; %.3f sec/batch)')

			print(format_str %(datetime.now(),self._step,loss_value,acc,
				examples_per_sec,sec_per_batch))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
if MODE == 'train':

	file_writer = tf.summary.FileWriter(LOGDIR,tf.get_default_graph())
	with tf.train.MonitoredTrainingSession(
			save_checkpoint_secs=70,
			checkpoint_dir=LOGDIR,
			hooks=[tf.train.StopAtStepHook(last_step=NUM_EPOCHS*NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN),
					tf.train.NanTensorHook(loss),
					_LoggerHook()],
			config=config) as mon_sess:
		original_saver.restore(mon_sess,INCEPTION_V3_CHECKPOINT)
		global_step = tf.contrib.framework.get_or_create_global_step()
		print("Proceeding to training stage")
		i = 0
		while not mon_sess.should_stop():
			#print(mon_sess.run(global_step))
			#print([mon_sess.run(v) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='flower_logits')])
			#mon_sess.run(accuracy,feed_dict={training:False})
			#print(mon_sess.run(a[0])[0][0])
			#print("before %f" %mon_sess.run(accuracy,feed_dict={training:False}))
			#mon(train_opn_sess.ru,feed_dict={training:True})
			mon_sess.run(train_op,feed_dict={training:True})
			print('acc: %f' %mon_sess.run(accuracy,feed_dict={training:False}))
			#print('loss: %f' %mon_sess.run(loss,feed_dict={training:False}))
			#print('passed')


elif MODE == 'test':
	init = tf.global_variables_initializer()
	ckpt = tf.train.get_checkpoint_state(LOGDIR)
	if ckpt and ckpt.model_checkpoint_path:
		with tf.Session(config=config) as sess:
				init.run()
				saver = tf.train.Saver()
				print(ckpt.model_checkpoint_path)
				saver.restore(sess,ckpt.model_checkpoint_path)
				global_step = tf.contrib.framework.get_or_create_global_step()
				#print(global_step.eval())
				print([(v.name,v.eval())  for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="flower_logits")])
				coord = tf.train.Coordinator()
				threads =[]
				try:
					for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
						threads.extend(qr.create_threads(sess, coord=coord, daemon=True,start=True))
					print('model restored')
					i =0
					num_iter = 4*NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/BATCH_SIZE
					print(num_iter)
					while not coord.should_stop() and i < num_iter:
						print("loss: %.2f," %loss.eval(feed_dict={training:True}),end="")
						print("acc: %.2f" %accuracy.eval(feed_dict={training:True}))
						i+=1
				except Exception as e:
					print(e)
					coord.request_stop(e)
				coord.request_stop()
				coord.join(threads,stop_grace_period_secs=10)