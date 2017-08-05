import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import numpy as np
from datetime import datetime


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

MODE = 'train'
HEIGHT= 32
WIDTH = 32
DEPTH = 3
BATCH_SIZE = 64
NUM_EPOCHS = 50
LOGDIR = '/tmp/gan'
CIFAR_DATA = '/tmp/cifar10_data/cifar-10-batches-bin'


NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
FRACTION_IN_QUEUE = 0.4
LOG_FREQUENCY = 10
lrg = 0.0002
lrd = 0.0002
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
	image = tf.reshape(image_part_linear,[DEPTH,HEIGHT,WIDTH])
	image = tf.transpose(image,[1,2,0])
	#image = tf.reshape(image_part_linear,[HEIGHT,WIDTH,DEPTH])
	return (key,image,label)

def format_input(image,label):
	image = tf.cast(image,tf.float32)
	image= tf.image.resize_images(image, [2*HEIGHT, 2*WIDTH], method=tf.image.ResizeMethod.BILINEAR)
	scaled = image/127.5 - 1.
	scaled.set_shape([2*HEIGHT,2*WIDTH,DEPTH])
	label.set_shape([1])
	#print(tf.reduce_max(image_norm))
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

	tf.summary.image('images_s',image_batch)
	#reshape to labels to vector
	return image_batch,tf.reshape(label_batch,[BATCH_SIZE])

#preprocessing and training
with tf.device('/cpu:0'):
	filenames = [CIFAR_DATA +('/data_batch_%d.bin' % i) 
                 for i in range(1, 6)]
	print('starting to load data. please wait...')
	key,image,label  = parse_data(filenames)
	scaled,label = format_input(image,label)
	image_batch,_ = gen_batch(scaled,label)

def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)
is_training = tf.placeholder(tf.bool, shape=(),name="is_training")
"""
'variables_collections': {
'beta': None,
'gamma': None,
'moving_mean': ['moving_vars'],
'moving_variance': ['moving_vars'],
}
"""
batch_norm_params = {
	'is_training':is_training,
	# Decay for the moving averages.
	'decay': 0.9,
	# epsilon to prevent 0s in variance.
	'epsilon': 1e-5,
	# collection containing update_ops.
	'updates_collections': None,
	'scale':True
	# collection containing the moving mean and moving variance.
  }

code_size = 100

#uniform noise distribution
Z = tf.placeholder(tf.float32,shape=(None,code_size),name="Z")


def bn(x,scope,reuse=False):
	return tf.contrib.layers.batch_norm(x,
	                      decay=0.9, 
	                      updates_collections=None,
	                      epsilon=1e-5,
	                      scale=True,
	                      reuse=reuse,
	                      is_training=is_training,
	                      scope=scope)


#project code into 4D tensor
code_projection = 4*4*512
"""
proj_z = slim.fully_connected(Z,code_projection,
	activation_fn=tf.nn.relu,
	normalizer_fn=slim.batch_norm,
	normalizer_params=batch_norm_params,
	weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
	scope="z_projection")
"""

proj_z = slim.fully_connected(Z,code_projection,
	activation_fn=tf.nn.relu,
	weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
	normalizer_fn=slim.batch_norm,
	normalizer_params=batch_norm_params,
	scope="z_projection")
z_reshape = tf.reshape(proj_z,[-1,4,4,512])
"""
relu = tf.nn.relu

z_reshape= relu(bn(z_reshape,scope="bn0"))
gen_scope = "generative"
conv1 = tf.nn.relu(bn(slim.conv2d_transpose(z_reshape,256,(5,5),2,activation_fn=None,scope="gen",weights_initializer=tf.truncated_normal_initializer(stddev=0.02)),"bn1"))
conv2 = tf.nn.relu(bn(slim.conv2d_transpose(conv1,    128,(5,5),2,activation_fn=None,scope="gen1",weights_initializer=tf.truncated_normal_initializer(stddev=0.02)),"bn2"))
conv3 = tf.nn.relu(bn(slim.conv2d_transpose(conv2,    64,(5,5), 2,activation_fn=None,scope="gen2",weights_initializer=tf.truncated_normal_initializer(stddev=0.02)),"bn3"))
conv4 = slim.conv2d_transpose(conv3,     3,(5,5), 2,scope="gen3",activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
gen = tf.nn.tanh(conv4)
sampler = (1.+gen)/2.
"""
gen_arch =[
			(256,(5,5),2,"SAME","NHWC",tf.nn.relu,slim.batch_norm,batch_norm_params),
			(128,(5,5),2,"SAME","NHWC",tf.nn.relu,slim.batch_norm,batch_norm_params),
			(64,(5,5),2,"SAME","NHWC",tf.nn.relu,slim.batch_norm,batch_norm_params),
			(3,(5,5),2,"SAME","NHWC",tf.nn.tanh),
		]

gen =slim.stack(z_reshape,slim.conv2d_transpose,gen_arch,
	weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
	scope="generative")

sampler = (1.+gen)/2
#tf.summary.image("sample",sampler)
"""
dis_conv1 = lrelu(slim.conv2d(image_batch,64,(5,5),2,activation_fn=None,scope="dis1",weights_initializer=tf.truncated_normal_initializer(stddev=0.02)))
dis_conv2 = lrelu(bn(slim.conv2d(dis_conv1, 128,(5,5),2,activation_fn=None,scope="dis2",weights_initializer=tf.truncated_normal_initializer(stddev=0.02)),scope="dbn1"))
dis_conv3 = lrelu(bn(slim.conv2d(dis_conv2, 256,(5,5),2,activation_fn=None,scope="dis3",weights_initializer=tf.truncated_normal_initializer(stddev=0.02)),scope="dbn2"))
dis_conv4 = lrelu(bn(slim.conv2d(dis_conv3, 512,(5,5),2,activation_fn=None,scope="dis4",weights_initializer=tf.truncated_normal_initializer(stddev=0.02)),scope="dbn3"))

disX = slim.fully_connected(dis_conv4,1,
	weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
	activation_fn=None,
	scope="dis_out")

dis_conv12 = lrelu(slim.conv2d(gen,64,(5,5),2,scope="dis1",activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),reuse=True))
dis_conv22 = lrelu(bn(slim.conv2d(dis_conv12, 128,(5,5),2,activation_fn=None,scope="dis2",weights_initializer=tf.truncated_normal_initializer(stddev=0.02),reuse=True),scope="dbn1",reuse=True))
dis_conv32 = lrelu(bn(slim.conv2d(dis_conv22, 256,(5,5),2,activation_fn=None,scope="dis3",weights_initializer=tf.truncated_normal_initializer(stddev=0.02),reuse=True),scope="dbn2",reuse=True))
dis_conv42 = lrelu(bn(slim.conv2d(dis_conv32, 512,(5,5),2,activation_fn=None,scope="dis4",weights_initializer=tf.truncated_normal_initializer(stddev=0.02),reuse=True),scope="dbn3",reuse=True))

disG = slim.fully_connected(dis_conv42,1,
	activation_fn=None,
	scope="dis_out",reuse=True)
"""



dis_arch = [

			(64,(5,5),2,"SAME","NHWC",1,lrelu),
			(128,(5,5),2,"SAME","NHWC",1,lrelu,slim.batch_norm,batch_norm_params),
			(256,(5,5),2,"SAME","NHWC",1,lrelu,slim.batch_norm,batch_norm_params),
			(512,(5,5),2,"SAME","NHWC",1,lrelu,slim.batch_norm,batch_norm_params),
			(1,  (1,1),1,"VALID","NHWC",1,None)
]

dis = "discriminative"
"""
disX = slim.stack(image_batch,slim.conv2d,dis_arch,
	weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
	scope=dis)

disG = slim.stack(gen,slim.conv2d,dis_arch,
	scope=dis,
	reuse=True)

disX = slim.fully_connected(disX,1,
	weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
	normalizer_fn=slim.batch_norm,
	normalizer_params=batch_norm_params,
	activation_fn=None,
	scope="dis_out")

disG = slim.stack(gen,slim.conv2d,dis_arch,
	#normalizer_fn=slim.batch_norm,
	#normalizer_params=batch_norm_params,
	scope=dis,
	reuse=True)

disG = slim.fully_connected(disG,1,
	normalizer_fn=slim.batch_norm,
	normalizer_params=batch_norm_params,
	activation_fn=None,
	scope="dis_out",
	reuse=True)
"""





gscopes=['generative','z_projection']#['gen','gen1','gen2','gen3','z_projection','bn1','bn2','bn3','bn0']
dscopes=[dis]#['dis1','dis2','dis3','dis4','dis_out','dbn1','dbn2','dbn3']

#import pdb;pdb.set_trace()

#prob = tf.reduce_mean(tf.nn.sigmoid(tf.squeeze(disG)))
global_step = tf.contrib.framework.get_or_create_global_step()
with tf.name_scope("dloss"):
		logits_real = disX#tf.squeeze(disX)
		logits_fake = disG#tf.squeeze(disG)
		dloss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real,labels=tf.ones_like(logits_real)))
		dloss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,labels=tf.zeros_like(logits_fake)))
		dloss= dloss_real + dloss_fake
		#loss_avg = tf.train.ExponentialMovingAverage(0.9,name='flower_avg')
		#get the moving average ops (create shadow variables)
		#loss_avg_op = loss_avg.apply([dloss])
		#log loss and shadow variables for avg loss
		#tf.summary.scalar(dloss.op.name+' (raw)',dloss)
		#tf.summary.scalar(dloss.op.name,loss_avg.average(dloss))
		var = [ tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=s) for s in dscopes]
		opt = tf.train.AdamOptimizer(lrd,beta1=0.5)
		#with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope=dis)):
		dtrain_op = opt.minimize(dloss,global_step=global_step,var_list=var)

with tf.name_scope("gloss"):

		gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,labels=tf.ones_like(logits_fake)))

		#loss_avg = tf.train.ExponentialMovingAverage(0.9,name='flower_avg')
		#get the moving average ops (create shadow variables)
		#loss_avg_op = loss_avg.apply([gloss])
		#log loss and shadow variables for avg loss
		#tf.summary.scalar(gloss.op.name+' (raw)',gloss)
		#tf.summary.scalar(gloss.op.name,loss_avg.average(gloss))
		var = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=s) for s in gscopes]
		opt = tf.train.AdamOptimizer(lrg,beta1=0.5)

		#with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope='generative')):
		gtrain_op = opt.minimize(gloss,global_step=global_step,var_list=var)


class _LoggerHook(tf.train.SessionRunHook):

	def begin(self):
		self._step = -1
		self._start_time = time.time()
	def before_run(self,run_context):
		self._step+=1
		return tf.train.SessionRunArgs([gloss,dloss])

	def after_run(self,run_context,run_values):
		#print(run_values.results[2])
		if self._step % LOG_FREQUENCY ==0:
			current_time = time.time()
			duration = current_time - self._start_time
			self._start_time = current_time
			"""
			gloss_value = run_values.results[0]
			dloss_value = run_values.results[1]
			dval = 0#run_values.results[2]
			examples_per_sec = LOG_FREQUENCY/duration
			sec_per_batch = duration / LOG_FREQUENCY

			format_str = ('%s: step %d, gloss = %.2f, dloss %.2f, dval %.2f (%.1f examples/sec; %.3f sec/batch)')

			print(format_str %(datetime.now(),self._step,gloss_value, dloss_value,dval,
				examples_per_sec,sec_per_batch))
			"""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
file_writer = tf.summary.FileWriter(LOGDIR,tf.get_default_graph())


def plot(samples):
	fig= plt.figure(figsize=(8,8))
	gs = gridspec.GridSpec(8,8)
	gs.update(wspace=0.05,hspace=0.05)

	for i,sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(np.floor(127.5*(sample.reshape(64,64,3))),cmap="Greys_r")
	return fig

import os
if not os.path.exists('gan2/'):
	os.makedirs('gan2/')

def plotfig(samples,i):
	fig = plot(samples)

	plt.savefig('gan2/{}.png'.format(str(i).zfill(3)),bbox_inches='tight')
	plt.close(fig)

with tf.train.MonitoredTrainingSession(
	save_checkpoint_secs=100,
	checkpoint_dir=LOGDIR,
	hooks=[
		tf.train.StopAtStepHook(last_step=NUM_EPOCHS*NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN),
		tf.train.NanTensorHook(dloss),
		tf.train.NanTensorHook(gloss),
		_LoggerHook()],
	config=config) as mon_sess:		
	print("Proceeding to training stage")
	i = 0
	while not mon_sess.should_stop():
		if i %100 == 0:

			plotfig(mon_sess.run(sampler,feed_dict={Z:np.random.uniform(-1.,1.,size=(64,code_size)).astype(np.float32),is_training:False}),i)
		break
		i+=1
		batch_z = np.random.uniform(-1.,1.,size=(BATCH_SIZE,code_size)).astype(np.float32)
		mon_sess.run(dtrain_op,feed_dict={Z:batch_z,is_training:True})
		mon_sess.run(gtrain_op,feed_dict={Z:batch_z,is_training:True})
		mon_sess.run(gtrain_op,feed_dict={Z:batch_z,is_training:True})
		if i %10 == 0:
			print(mon_sess.run(dloss,feed_dict={Z:batch_z,is_training:False}))
			print(mon_sess.run(gloss,feed_dict={Z:batch_z,is_training:False}))


"""
print("starting")
init = tf.global_variables_initializer()
i = 0
print(tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS))
with tf.Session(config=config) as mon_sess:
	init.run()

	coord = tf.train.Coordinator()
	threads =[]
	for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
		threads.extend(qr.create_threads(mon_sess, coord=coord, daemon=True,start=True))

	while i < NUM_EPOCHS and not coord.should_stop():
		print("WTD")
		try:
			if i %20 == 0:
				plotfig(mon_sess.run(sampler,feed_dict={Z:np.random.uniform(-1.,1.,size=(16,code_size)).astype(np.float32),is_training:False}),i)
			i+=1
			batch_z = np.random.uniform(-1.,1.,size=(BATCH_SIZE,code_size)).astype(np.float32)
			mon_sess.run(dtrain_op,feed_dict={Z:batch_z,is_training:True})
			mon_sess.run(gtrain_op,feed_dict={Z:batch_z,is_training:True})
			mon_sess.run(gtrain_op,feed_dict={Z:batch_z,is_training:True})
			if i %10 == 0:
				print(mon_sess.run(dloss,feed_dict={Z:batch_z,is_training:False}))
				print(mon_sess.run(gloss,feed_dict={Z:batch_z,is_training:False}))
		except Exception as e:
			print(e)
			break
	coord.request_stop()
	coord.join(threads,stop_grace_period_secs=10)

"""


