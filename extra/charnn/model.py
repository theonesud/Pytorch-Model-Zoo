import tensorflow as tf


class CharRNN:

    # Hyperparameters
    no_seqs = 256  # Try 128
    seq_len = 256
    no_hidden = 512
    no_layers = 2  # Try 3
    keep_prob = 0.5
    learning_rate = 0.0005
    grad_clip = 5

    def __init__(self, no_classes, sampling=False):

        if sampling:
            self.no_seqs, self.seq_len, self.keep_prob = 1, 1, 1

        # Input layer
        self.inputs = tf.placeholder(tf.int32, [self.no_seqs, self.seq_len])
        x_one_hot = tf.one_hot(self.inputs, no_classes)

        # LSTM layer
        lstm = tf.contrib.rnn.BasicLSTMCell(self.no_hidden)
        drop = tf.contrib.rnn.DropoutWrapper(
            lstm, output_keep_prob=self.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([drop] * self.no_layers)

        self.initial_state = cell.zero_state(self.no_seqs, tf.float32)

        lstm_outputs, self.final_state = tf.nn.dynamic_rnn(
            cell, x_one_hot, initial_state=self.initial_state)

        lstm_outputs = tf.concat(lstm_outputs, axis=1)
        lstm_outputs = tf.reshape(lstm_outputs, [-1, self.no_hidden])

        # Softmax layer
        # (Need to put them in a different scope because tf.contrib.rnn makes
        #  weights and biases too, with the default scope)
        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal(
                (self.no_hidden, no_classes), stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(no_classes))

        logits = tf.matmul(lstm_outputs, softmax_w) + softmax_b
        self.prediction = tf.nn.softmax(logits)

        # Calculate the loss
        self.targets = tf.placeholder(tf.int32, [self.no_seqs, self.seq_len])
        y_one_hot = tf.one_hot(self.targets, no_classes)
        y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y_reshaped)
        self.loss = tf.reduce_mean(loss)

        # Define the training operation
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
