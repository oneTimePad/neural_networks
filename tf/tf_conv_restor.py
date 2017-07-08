import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
#change
mnist = input_data.read_data_sets("C:\\Users\\MTechLap\\Downloads\\mnist-train")

from datetime import datetime
now  = datetime.utcnow().strftime(("%Y%m%d%H%M%S"))
root_logdir = "tf_logs"
logdir = "{}/run-{}".format(root_logdir,now)




config = tf.ConfigProto()
config.gpu_options.allow_growth = True
batch_size = 100
#restore and test
with tf.Session(config=config) as sess:
	saver = tf.train.import_meta_graph("./my_mnist_model.meta")
	#file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())
	saver.restore(sess,"./my_mnist_model")
	
	graph  =tf.get_default_graph()
	fc5 = graph.get_tensor_by_name("fc5/BiasAdd:0")
	logits = tf.reshape(fc5,shape=[-1,10])							

	y = graph.get_tensor_by_name("inputs/y:0")
	with tf.name_scope("restore_eval"):
		correct = tf.nn.in_top_k(logits,y,1)
		accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
	for iteration in range(mnist.test.num_examples // batch_size):
		X_batch,y_batch = mnist.test.next_batch(batch_size)
		acc_test  = accuracy.eval(feed_dict={"inputs/X:0": X_batch, "inputs/y:0": y_batch})
		print(acc_test)