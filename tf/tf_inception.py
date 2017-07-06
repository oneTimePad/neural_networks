import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import tensorflow.slim as slim
HEIGHT= 299
WIDTH = 299
DEPTH = 3
BATCH_SIZE = 128
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500
FRACTION_IN_QUEUE = 0.4
LOGDIR = "/tmp/inception"
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

X = tf.placeholder(tf.float32,shape=[None,299,299,3])
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits,end_points = inception.inception_v3(
                    X,num_classes=1001,is_training=False)
    saver = tf.Saver()
    saver.restore(sess,INCEPTION_V3_CHECKPOINT)
    file_writer = tf.summary.FileWriter(LOGDIR,tf.get_default_graph())
