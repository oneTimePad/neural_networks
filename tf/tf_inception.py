import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import tensorflow.slim as slim

"""
Inception transfer learning for flower data set based on
https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10
and
https://github.com/ageron/handson-ml/blob/master/13_convolutional_neural_networks.ipynb
"""


HEIGHT= 299
WIDTH = 299
DEPTH = 3
BATCH_SIZE = 128
NUM_EPOCHS = 100
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500
FRACTION_IN_QUEUE = 0.4
NUM_EPOCHS_PER_DECAY=1000
NUM_OUT = 5
INITIAL_LEARNING_RATE = 0.1
DECAY_RATE = 0.96
LOGDIR = '/tmp/inception'
FLOWERS_DIR = ''
def parse_data(filenames_queue):
    """
    args:
        filesname_queue (input)
    returns:
        key:= file key from reader
        image:= parsed images from reader
        label:= parsed label from reader
    """
    label_byte = 1
    image_bytes = WIDTH*HEIGHT*DEPTH
    #format for flower images is like cifar10
    #(label_byte)+linear_image
    record_bytes = label_bytes + image_bytes
    #read record bytes each for each sample
    reader = tf.FixedLengthReader(record_bytes=record_bytes)
    #unique entry identifier, value(string)
    key,value = reader.read(filenames_queue)
    #convert to bytes
    record_bytes = tf.decode_raw(value,tf.uint8)
    #extract label
    label = tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes],tf.int32)

    image_part_linear = tf.strided_slice(record_bytes,[label_bytes],[label_bytes+image_bytes])
    image = tf.reshape(image_part_linear,[DEPTH,HEIGHT,WIDTH])
    image = tf.transpose(image,[1,2,0])
    return (key,image,label)

def format_input(image,label):
    #add some randomness
    flip = tf.image.random_flip_left_right(image)
    bright_change = tf.image.random_brightness(flip,max_delta=63)
    #inception expect inputs [-1.0,1.0]
    scaled = tf.multipy(bright_change,2.0/255.0)
    scaled = tf.subtract(scaled,1,0)
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
    key,image,label  = parse_data(FLOWERS_DIR)
    scaled,label = format_input(image,label)
    image_batch,label_batch = gen_batch(scaled,label)


with slim.arg_scope(inception.inception_v3_arg_scope()):
    old_logits,end_points = inception.inception_v3(
                    image_batch,num_classes=1001,is_training=False)
with tf.name_scope('flower_output'):
    pre_logits = end_points['PreLogits']
    logits = tf.layers.conv2d(pre_logits,kernel_size=1,strides=1,filters=5,padding='SAME',name='flower_logits')
    prediction = tf.nn.softmax(logits,name='flower_softmax')
    #logits_full = tf.reshape(logits,shape=[-1,NUM_OUT])
    logits_full = tf.squeeze(logits)

with tf.name_scope('flower_loss'):
    label64 = tf.cast(label_batch,tf.int64)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label64,logits =logits_full)
    loss = tf.reduce_mean(xentropy,name='flower_loss)
    #get moving average
    loss_avg = tf.train.ExponentialMovingAverage(0.9,name='flower_avg')
    #get the moving average ops (create shadow variables)
    loss_avg_op = loss_avg.apply([loss])
    #log loss and shadow variables for avg loss
    tf.summary.scalar(loss.op.name+' (raw)',loss)
    tf.summary.scalar(loss.op.name,loss_avg.average(loss))

with tf.name_scope('flower_train'):
    global_step = tf.Variable(0,trainable=False)
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
        sgd_op = tf.train.GradientDescentOptimizer(lr,var_list=train_vars)
        grads = sgd_op.compute_gradients(loss)
    apply_grads_op = sgd.apply_gradients(grads,global_step=global_step)
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

class _LoggerHook(tf.train.SessionRunHook):

    def begin(self):
        self._step = -1
        self._start_time = time.time()
    def before_run(self,run_context):
        self._step+=1
        return tf.train.SessionRunArgs(loss)

    def after_run(self,run_context,run_values):
        if self._step % LOG_FREQUENCY == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time

            loss_value = run_values.results
            examples_per_sec = LOG_FREQUENCY/duration
            sec_per_batch = duration / LOG_FREQUENCY

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')

            print(format_str %(datetime.now(),self_step,loss_value,
                examples_per_sec,sec_per_batch))

saver = tf.Saver()
config = tf.ConfigProto()
file_writer = tf.summary.FileWriter(LOGDIR,tf.get_default_graph())
with tf.train.MonitoredTrainingSession(
        checkpoint_dir=LOG_DIR,
        hooks=[tf.train.StopAtStepHook(last_step=NUM_EPOCHS*NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN),
                tf.train.NanTensorHook(loss),
                _LoggerHook()],
        config=config) as mon_sess:
    saver.restore(sess,INCEPTION_V3_CHECKPOINT)
    while not mon_ses.should_stop():
        mon_sess.run(train_op)
