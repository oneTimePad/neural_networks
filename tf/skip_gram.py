

class NeuralNGram(object):


    def __init__(self, skip_window):
        """Implements the Skip-Gram model
        with vanilla Softmax
        """

        self.__skip_window = skip_window
        self.__batch_size = batch_size

    def __get_raw_data(self, data_path):
          """Load PTB raw data from data directory "data_path".
          Reads PTB text files, converts strings to integer ids,
          and performs mini-batching of the inputs.
          The PTB dataset comes from Tomas Mikolov's webpage:
          http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
          Args:
            data_path: string path to the directory where simple-examples.tgz has
              been extracted.
          Returns:
            tuple (train_data, valid_data, test_data, vocabulary)
            where each of the data objects can be passed to PTBIterator.
          """

          train_path = os.path.join(data_path, "ptb.train.txt")
          valid_path = os.path.join(data_path, "ptb.valid.txt")
          test_path = os.path.join(data_path, "ptb.test.txt")

          word_to_id = _build_vocab(train_path)
          train_data = _file_to_word_ids(train_path, word_to_id)
          valid_data = _file_to_word_ids(valid_path, word_to_id)
          test_data = _file_to_word_ids(test_path, word_to_id)
          vocabulary = len(word_to_id)
          return train_data, valid_data, test_data, vocabulary

    def __produce(self, raw_data):
        pass

    def generate_batch(self, batch_size, num_skips, skip_window):
        assert batch_size % num_skips  == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape = (batch_size), dtype = np.int32)
        labels = np.ndarray(shape = (batch_size, 1), dtype = np.int32)

        span = 2 * skip_window + 1
        buffer = collections.deque(maxlen = span)

        if self.__data_index + span > len(data):
            data_index = 0
        buffer.extend(data[data_index: data_index + span])
        self.__data_index += span

        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)

            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]
            if data_index == len(data):
                buffer.extend(self.__data[0:span])
                data_index = span
            else:
                buffer.append(self.__data[data_index])
                data_index += 1
        self.data_index = (data_index + len(data) - span) % len(data)
        return batch, labels

    def forward(self, examples, labels):
        """Builds forward pass for language model
        """
        init_width = 0.5 / self.__emb_dim

        # create input embeddings for context of sentence
        inpt_emb = tf.Variable(
            tf.random_uniform(
                [self.__vocab_size, self.__emb_dim], -init_width, init_width))

        self.__input_emb = inpt_emb

        output_emb_w = tf.Variable(
            tf.zeros([self.__vocab_size, self.__emb_dim]), name = "output_emb_w")

        output_emb_b = tf.Variable(
            tf.zeros([self.__vocab_size, self.__emb_dim]), name = "output_emb_b")

        self.__output_emb_w = output_emb_w
        self.__output_emb_b = output_emb_b

        sampled_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes = labels_matrix,
            num_true = 1,
            num_sampled = self.__num_neg_samples,
            unique = True,
            range_max = self.__vocab_size,
            distortion = 0.75
            unigrams = self.__vocab_counts.tolist())

        # get input embeddings
        examples_emb = tf.nn.embedding_lookup(input_emb, examples)

        true_w = tf.nn.embedding_lookup(output_emb_w, labels)
        true_b = tf.nn.embedding_lookup(output_emb_b, labels)

        sampled_w = tf.nn.embedding_lookup(output_emb_w, sampled_ids)
        sampled_b = tf.nn.embedding_lookup(output_emb_b, sampled_ids)

        # compute single unit
        true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b

        sampled_b_vec = tf.reshape(sampled_b, [self.__num_neg_samples])
        sampled_logits = tf.matmul(example_emb,
                                   sampled_w,
                                   transposed_b = True) + sampled_b_vec

        return true_logits, sampled_logits

    def nce_loss(self, true_logits, sampled_logits):
        """Build the graph for NCE loss"""

        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(true_logits), logits = true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(sampled_logits), logits = sampled_logits)

        nce_loss_tensor = (tf.reduce_sum(true_xent) + tf.reduce_sum(sampled_xent)) / self.__batch_size

        return nce_loss_tensor



    def run(self, data_path):

        train_data, valid_data, test_data, vocabulary = self.__get_raw_data(data_path)
