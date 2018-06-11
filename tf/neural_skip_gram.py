from collections import Counter, deque
import tensorflow as tf
import numpy as np
import random
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops import lookup_ops
import os
import codecs


UNK_ID = 0

class NeuralSkipGram(object):

    def __init__(self, filename,
                       logdir,
                       batch_size,
                       skip_window,
                       num_skips,
                       emb_dim,
                       num_samples,
                       init_lr,
                       num_steps):
        """Neural Language Model for Word2Vec Skip-gram
        """

        self.filename = filename
        self.logdir = logdir
        self.batch_size = batch_size
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.emb_dim = emb_dim
        self.num_samples = num_samples
        self.init_lr = init_lr
        self.num_steps = num_steps

    def __read_words(self, filename):
        with tf.gfile.GFile(filename, "r") as f:
            return f.read().replace("\n", "<eos>").split()

    def __parse_words(self, word_list):
        counts = Counter(word_list)

        self.words_to_ids = {w: i for i, w in enumerate(counts.keys())}
        self.reverse_dictionary = {i: w  for w, i in self.words_to_ids.items()}
        self.ids_to_counts = list(counts.values())
        self.data = [self.words_to_ids[w] for w in word_list]
        self.vocab_size = len(self.ids_to_counts)
        self.data_index = 0

    def generate_batch_dataset(self, corpus_file,
                                     vocab_file = None):
        # create vocab file, if doesn't exist
        if not vocab_file:
            with tf.gfile.GFile(corpus_file, "r") as f:
                corpus = f.read().replace("\n", "<eos>").split()
                counts = Counter(corpus)

                vocab_file = os.path.join(os.path.dirname(corpus_file), "vocab_file")
                with codecs.getwriter("utf-8")(tf.gfile.GFile(vocab_file, "wb")) as vocab:
                        for word in counts.keys():
                            vocab.write("%s\n" % word)
        span = 3

        vocab_table = lookup_ops.index_table_from_file(vocab_file,
                                                       default_value = UNK_ID)
        eos_id = tf.cast(vocab_table.lookup(tf.constant("<eos>")), tf.int64)
        dataset = tf.data.TextLineDataset(corpus_file)

        dataset = dataset.map(lambda string: tf.string_split([string]).values)

        dataset = dataset.map(lambda src : vocab_table.lookup(src))
        #dataset = dataset.padded_batch(1, padded_shapes=(tf.TensorShape([None])), padding_values=(eos_id)).prefetch(10)
        dataset = dataset.apply(tf.contrib.data.sliding_window_batch(window_size = 2, stride = 1))
        #import pdb;pdb.set_trace()
        #

        iter = dataset.make_initializable_iterator()
        d = iter.get_next()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(tf.tables_initializer())
            sess.run(init)
            sess.run(iter.initializer)

            print(sess.run(d))


    def generate_batch(self, skip_window, num_skips):
        batch = np.ndarray(shape = self.batch_size, dtype = np.int32)
        labels = np.ndarray(shape = self.batch_size, dtype = np.int32)

        span = 2 * skip_window + 1
        buffer = deque(maxlen = span)
        buffer.extend(self.data[self.data_index: self.data_index + span])
        for i in range(self.batch_size // num_skips):
            words_to_check = [ w for w in range(span) if w != skip_window]
            samples = random.sample(words_to_check, num_skips)
            for j, w in enumerate(samples):
                batch[ i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j] = buffer[w]

            if self.data_index == len(self.data):
                self.data_index = span
                buffer.extend(self.data[0:span])
            else:
                self.data_index += 1
                buffer.append(self.data[self.data_index])

        return batch, labels

    def forward(self, examples, labels):
        """forward pass of neural language model
        """
        init_width = 0.5/self.emb_dim
        input_emb = tf.Variable(
            tf.random_uniform(
                [self.vocab_size, self.emb_dim], -init_width, init_width), name = "input_embedding")
        self.input_emb = input_emb

        output_emb_w = tf.Variable(
            tf.zeros(
                [self.vocab_size, self.emb_dim]), name = "output_embedding_w")
        self.output_emb_w = output_emb_w

        output_emb_b = tf.Variable(
            tf.zeros(
                [self.vocab_size]), name = "output_embedding_b")
        self.output_emb_b = output_emb_b

        # [batch_size, emb_dim]
        batch_emb = tf.nn.embedding_lookup(input_emb, examples)

        # [batch_size, emb_dim]
        true_label_emb_w = tf.nn.embedding_lookup(output_emb_w, labels)
        # [batch_size]
        true_label_emb_b = tf.nn.embedding_lookup(output_emb_b, labels)

        true_label_logits = tf.reduce_sum(tf.multiply(batch_emb, true_label_emb_w), 1) + true_label_emb_b

        # the unigram sampler expects the true labels in this shape
        labels_matrix = tf.reshape(
            tf.cast(labels,
                    dtype = tf.int64),
            [self.batch_size, 1])

        # sample negatives from unigram distribution, based on word frequencies
        sampled_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes = labels_matrix,
            num_true = 1,
            num_sampled = self.num_samples,
            unique = True,
            range_max = self.vocab_size,
            distortion = 0.75,
            unigrams = self.ids_to_counts)

        # [num_samples, emb_dim]
        neg_emb_w = tf.nn.embedding_lookup(output_emb_w, sampled_ids)
        # [num_samples]
        neg_emb_b = tf.nn.embedding_lookup(output_emb_b, sampled_ids)

        neg_logits = tf.matmul(batch_emb, neg_emb_w, transpose_b = True) + neg_emb_b

        self.global_step = tf.Variable(0, name = "global_step")

        return true_label_logits, neg_logits

    def neg_loss(self, true_logits, neg_logits):
        """Computes the Negative Sampling loss
           Negative sampling approximates negative term by 1,
           this results in the use of a sigmoid for binary classification
        """

        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels = tf.ones_like(true_logits), logits = true_logits)

        neg_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels = tf.zeros_like(neg_logits), logits = neg_logits)

        return (tf.reduce_sum(true_xent) + tf.reduce_sum(neg_xent)) / self.batch_size

    def optimize(self, loss):
        """Create Opitmizer
        """

        optimizer = tf.train.AdamOptimizer(self.init_lr)
        train = optimizer.minimize(loss, global_step = self.global_step, gate_gradients = optimizer.GATE_NONE)

        self.train_op = train

    def build_graph(self):
        words_list = self.__read_words(self.filename)
        self.__parse_words(words_list)
        with tf.name_scope("inputs"):
            examples = tf.placeholder(tf.int32, shape = [self.batch_size], name = "examples")
            labels = tf.placeholder(tf.int32, shape = [self.batch_size], name = "labels")

        true_logits, neg_logits = self.forward(examples, labels)
        total_loss = self.neg_loss(true_logits, neg_logits)

        self.optimize(total_loss)
        self.loss = total_loss

        tf.summary.scalar("loss", self.loss)

        self.merged = tf.summary.merge_all()

    def train(self):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(self.logdir, sess.graph)

            sess.run(init)

            run_metadata = tf.RunMetadata()

            for step in range(self.num_steps):

                batch, labels = self.generate_batch(self.skip_window, self.num_skips)

                _, summary = sess.run(
                     [self.train_op,self.merged],
                     feed_dict = {"inputs/examples:0" : batch, "inputs/labels:0" : labels},
                     run_metadata = run_metadata)

                writer.add_summary(summary, step)

                if step == (self.num_steps - 1):
                    writer.add_run_metadata(run_metadata, "step%d" % step)
            with open(self.logdir + '/metadata.tsv', 'w') as f:
                for i in range(self.vocab_size):
                    f.write(self.reverse_dictionary[i] + '\n')
            saver.save(sess, os.path.join(self.logdir, "model.ckpt"))
            config = projector.ProjectorConfig()
            embedding_conf = config.embeddings.add()
            embedding_conf.tensor_name = self.input_emb.name
            embedding_conf.metadata_path = os.path.join(self.logdir, "metadata.tsv")
            projector.visualize_embeddings(writer, config)
        writer.close()


skip_gram = NeuralSkipGram("/home/lie/models/tutorials/embedding/text8",
                          "/tmp/word2vec",
                            batch_size = 128,
                            skip_window = 1,
                            num_skips = 2,
                            emb_dim = 128,
                            num_samples = 64,
                            init_lr = 0.001,
                            num_steps = 200000)
skip_gram.generate_batch_dataset("/home/lie/nmt/nmt/scripts/iwslt15/train.en")
#skip_gram.build_graph()
#skip_gram.train()
