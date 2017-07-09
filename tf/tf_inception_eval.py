





ckpt = tf.train.get_checkpoint_state(LOGDIR)
if chkpt and ckpt.model_checkpoint_path:
	with tf.Session(config=config) as sess:
			saver = tf.train.Saver()
			saver.restore(sess,model_checkpoint_path)

			print([(v.name,v.eval())  for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="flower_logits")])
			coord = tf.train.Coordinator()
			threads =[]
			try:
				for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
					threads.extend(qr.create_threads(sess, coord=coord, daemon=True,start=True))
				print('model restored')
				i =0
				num_iter = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/BATCH_SIZE
				while not coord.should_stop() and i < num_iter:
					print("loss: %.2f," %loss.eval(feed_dict={training:False}),end="")
					print("acc: %.2f" %accuracy.eval(feed_dict={training:False}))
					i+=1
			except Exception as e:
				print(e)
				coord.request_stop(e)
			coord.request_stop()
			coord.join(threads,stop_grace_period_secs=10)}}
