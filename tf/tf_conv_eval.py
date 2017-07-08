import tensorflow as tf


with tf.Session() as sess
	ckpt = tf.train.get_checkpoint_state("/tmp/cifar10_train7")
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess,ckpt.model_checkpoint_path)
		global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
	else:
		print("No Checkpoint file found")
		return
	
	coord = tf.train.Coordinator()
	try:
		threads = []
		for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
			threads.extend(qr.create_threads(sess,coord=coord,daemon=True,
								start=True))
		num_iter = int(math.ceil(NUM_EXAMPLES/BATCH_SIZE))
		true_count = 0
		total_sample_count = num_iter*BATCH_SIZE
		step = 0
		while strp<num_iter and not coord.should_stop():
			predictions = sess.run([top_k_op])
			true_count += np.sum(predictions)
			step+=1
		precision = true_count/total_sample_count
		print("%s: precision @1 =%.3f" %(datetime.now(),precision))
		summary = tf.Summary()
		summary.ParseFromString(sess.run(summary_op))
		summary.value.add(tag="Precision @ 1",simple_value=precision)
		summary_writer.add_summary(summary,global_step)
	except Exception as e:
		coord.request_stop(e)
	coord.request_stop()
	coord.join(threads,stop_grace_period_secs=10)

	