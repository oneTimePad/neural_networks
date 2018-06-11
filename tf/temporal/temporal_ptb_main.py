import tensorflow as tf
import tensorflow.contrib.slim as slim
from temporal_conv_net import TemporalConvNet
import reader

def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def main():
    vocab_size = 10000
    hidden_size = 200
    is_training = True
    kernel_size = 3
    batch_size = 20
    num_steps = 20
    lr = 0.01
    levels = 6
    dropout_prob = 0.2
    RAW_DATA_PATH = "/home/lie/lstm/data"
    num_epochs = 10
    num_steps = 10

    raw_data = reader.ptb_raw_data(RAW_DATA_PATH)
    train_data, valid_data, test_data, _ = raw_data
    input_data, targets = reader.ptb_producer(train_data, batch_size, num_steps)







    with tf.device("/cpu:0"):
        embedding = tf.get_variable(
            "embedding", [vocab_size, hidden_size], dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, input_data)


    temporal_network = TemporalConvNet([hidden_size] * levels,
                                        kernel_size,
                                        dropout_prob)(inputs, is_training)
    temporal_network = tf.reshape(temporal_network, [batch_size, -1])
    outputs = temporal_network

    def make_cell():
        cell = tf.contrib.rnn.BasicLSTMCell(
            hidden_size, forget_bias = 0.0, state_is_tuple = True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = 0.2)
        return cell
    cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(2)])
    state = cell.zero_state(batch_size, tf.float32)
    outputs = []
    with tf.variable_scope("RNN"):
        for time_step in range(num_steps):
            (cell_output, state) = cell(inputs[:, time_step, :], state)
            outputs.append(cell_output)
    outputs = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
    """
    import pdb;pdb.set_trace()
    logits = slim.fully_connected(outputs, vocab_size * num_steps , activation_fn = None)
    #logits = slim.fully_connected(outputs, vocab_size , activation_fn = None)


    logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])

    loss = tf.contrib.seq2seq.sequence_loss(logits,
                                            targets,
                                            tf.ones([batch_size, num_steps], dtype = tf.float32),
                                            average_across_timesteps = False,
                                            average_across_batch = True)


    cost = tf.reduce_sum(loss)
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(cost)
    init = tf.global_variables_initializer()
    queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
    coord = tf.train.Coordinator()
    with tf.Session() as sess:
        sess.run(init)
        t = []

        for th in queue_runners:
            t.extend(th.create_threads(sess, coord=coord, daemon=True, start=True))
        for epoch in range(num_epochs):
            for step in range(num_steps):
                print("Train")
                sess.run(train_op)
                print(sess.run(cost))





main()

"""
class PTBModel(object)

    def __init__(self, is_training, config, input_):
        self._is_training = is_training
        self._input = input_
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps

        size = config.hidden_size
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
        levels = config.levels
        dropout_prob = config.dropout_prob
        kernel_size = config.kernel_size



        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        temporal_network = TemporalConvNet([hidden_size] * levels,
                                            kernel_size,
                                            dropout_prob)(inputs, is_training)

        logits = slim.fully_connected(temporal_network, vocab_size * self.num_steps , activation_fn = None )

        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        loss = tf.contrib.seq2seq.sequence_loss(logits,
                                                input_.targets,
                                                tf.ones([self.batch_size, self.num_steps], dtype = data_type()),
                                                average_across_timsteps = False,
                                                average_across_batch = True)


        self._cost = tf.reduce_sum(loss)
        if not is_training:
            return

        optimizer = tf.train.AdamOptimizer(self._lr)
        optimizer.minimize(self._cost)
"""
