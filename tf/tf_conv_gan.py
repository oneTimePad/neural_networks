import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import numpy as np
from datetime import datetime


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

"""
DCGAN for Cifar10 inspired by https://github.com/carpedm20/DCGAN-tensorflow and https://github.com/fzliu/tf-dcgan 
and the TF Cifar10 tutorial

"""



HEIGHT= 32
WIDTH = 32
DEPTH = 3
BATCH_SIZE = 64
NUM_EPOCHS = 50
LOGDIR = '/tmp/gan60'
CIFAR_DATA = '/tmp/cifar10_data/cifar-10-batches-bin'
SAMPLES_DIR = '/tmp/gan_samples/'
SAMPLE_STEP = 100
SAMPLES = 16
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
FRACTION_IN_QUEUE = 0.4
LOG_FREQUENCY = 10
lrg = 0.0002
lrd = 0.0002
code_size = 100
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

	tf.summary.image('images',image_batch)
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

#not used 
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)
is_training = tf.placeholder(tf.bool, shape=(),name="is_training")

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


Z = tf.placeholder(tf.float32,shape=(None,1,1,code_size),name="Z")

gen_arch =[
			(512,(4,4),1,"VALID","NHWC",tf.nn.relu,slim.batch_norm,batch_norm_params),
			(256,(5,5),2,"SAME","NHWC",tf.nn.relu,slim.batch_norm,batch_norm_params),
			(128,(5,5),2,"SAME","NHWC",tf.nn.relu,slim.batch_norm,batch_norm_params),
			(64,(5,5),2,"SAME","NHWC",tf.nn.relu,slim.batch_norm,batch_norm_params),
			(3,(3,3),2,"SAME","NHWC",tf.nn.tanh),
		]

gen =slim.stack(Z,slim.conv2d_transpose,gen_arch,
	weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
	scope="generative")
sampler = (1.+gen)/2

dis_arch = [
			(128,(5,5),2,"SAME","NHWC",1,tf.nn.relu),
			(256,(5,5),2,"SAME","NHWC",1,tf.nn.relu,slim.batch_norm,batch_norm_params),
			(512,(5,5),2,"SAME","NHWC",1,tf.nn.relu,slim.batch_norm,batch_norm_params),
			(1024,(5,5),2,"SAME","NHWC",1,tf.nn.relu,slim.batch_norm,batch_norm_params),
			(1,  (1,1),1,"SAME","NHWC",1,None)
]



disX = slim.stack(image_batch,slim.conv2d,dis_arch,
	weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
	scope="discriminative")

disG = slim.stack(gen,slim.conv2d,dis_arch,
	weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
	scope="discriminative",
	reuse=True)


gscopes=['generative']
dscopes=['discriminative']


global_step = tf.contrib.framework.get_or_create_global_step()
with tf.name_scope("dloss"):
		logits_real = disX#tf.squeeze(disX)
		logits_fake = disG#tf.squeeze(disG)
		dloss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real,labels=tf.ones_like(logits_real)))
		dloss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,labels=tf.zeros_like(logits_fake)))
		dloss= dloss_real + dloss_fake

		var = [ tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=s) for s in dscopes]
		opt = tf.train.AdamOptimizer(lrd,beta1=0.5)

		dtrain_op = opt.minimize(dloss,global_step=global_step,var_list=var)

with tf.name_scope("gloss"):

		gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,labels=tf.ones_like(logits_fake)))

		var = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=s) for s in gscopes]
		opt = tf.train.AdamOptimizer(lrg,beta1=0.5)

		gtrain_op = opt.minimize(gloss,global_step=global_step,var_list=var)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
file_writer = tf.summary.FileWriter(LOGDIR,tf.get_default_graph())


def plot(samples):
	fig= plt.figure(figsize=(4,4))
	gs = gridspec.GridSpec(4,4)
	gs.update(wspace=0.05,hspace=0.05)

	for i,sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow((sample.reshape(64,64,3)),cmap="Greys_r")
	return fig

import os
if not os.path.exists(SAMPLES_DIR):
	os.makedirs(SAMPLES_DIR)

def plotfig(samples,i):
	fig = plot(samples)

	plt.savefig(SAMPLES_DIR+'{}.png'.format(str(i).zfill(3)),bbox_inches='tight')
	plt.close(fig)

with tf.train.MonitoredTrainingSession(
	save_checkpoint_secs=100,
	checkpoint_dir=LOGDIR,
	hooks=[
		tf.train.StopAtStepHook(last_step=NUM_EPOCHS*NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN),
		tf.train.NanTensorHook(dloss),
		tf.train.NanTensorHook(gloss)],
	config=config) as mon_sess:		
	print("Proceeding to training stage")
	step = 0
	while not mon_sess.should_stop():
		if step %SAMPLE_STEP == 1:
			plotfig(mon_sess.run(sampler,feed_dict={Z:np.random.uniform(-1.,1.,size=(SAMPLES,1,1,code_size)).astype(np.float32),is_training:False}),step)
		step+=1
		batch_z = np.random.uniform(-1.,1.,size=(BATCH_SIZE,1,1,code_size)).astype(np.float32)
		mon_sess.run(dtrain_op,feed_dict={Z:batch_z,is_training:True})
		mon_sess.run(gtrain_op,feed_dict={Z:batch_z,is_training:True})
		mon_sess.run(gtrain_op,feed_dict={Z:batch_z,is_training:True})
		if step %LOG_FREQUENCY==0:
			dloss_val,gloss_val = mon_sess.run([dloss,gloss],feed_dict={is_training:False,Z:batch_z})
			print("Gloss: %f, Dloss %f" %(dloss_val, gloss_val))
			

