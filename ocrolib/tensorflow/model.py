import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, DropoutWrapper
from tensorflow.python.ops import ctc_ops
import numpy as np


class Model:

    @staticmethod
    def load(filename, learning_rate=1e-3):
        print("Loading tensorflow model from root %s" % filename)
        graph = tf.Graph()
        with graph.as_default() as g:
            session = tf.Session(graph=graph)
            with tf.variable_scope("", reuse=False) as scope:

                saver = tf.train.import_meta_graph(filename + '.meta')
                saver.restore(session, filename)

                inputs = g.get_tensor_by_name("inputs:0")
                seq_len = g.get_tensor_by_name("seq_len:0")
                targets = tf.SparseTensor(
                    g.get_tensor_by_name("targets/indices:0"),
                    g.get_tensor_by_name("targets/values:0"),
                    g.get_tensor_by_name("targets/shape:0"))
                cost = g.get_tensor_by_name("cost:0")
                optimizer = g.get_operation_by_name('optimizer')
                ler = g.get_tensor_by_name("ler:0")
                decoded = g.get_tensor_by_name("decoded:0")
                logits = g.get_tensor_by_name("softmax:0")

                return Model(graph, session, inputs, seq_len, targets, optimizer, cost, ler, decoded, logits)

    @staticmethod
    def create(num_features, num_hidden, num_classes, num_layers=1, learning_rate=1e-3, reuse_variables=False):
        graph = tf.Graph()
        with graph.as_default():
            session = tf.Session(graph=graph)

            inputs = tf.placeholder(tf.float32, shape=(None, None, num_features), name="inputs")
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.placeholder(tf.int32, shape=(None,), name="seq_len")
            targets = tf.sparse_placeholder(tf.int32, shape=(None, None), name="targets")

            with tf.variable_scope("", reuse=reuse_variables) as scope:

                def get_lstm_cell(use_peepholse=False):
                    return LSTMCell(num_hidden, use_peepholes=use_peepholse, reuse=reuse_variables)

                rnn_inputs = inputs
                for i in range(num_layers):
                    fw, bw = get_lstm_cell(), get_lstm_cell()
                    (output_fw, output_bw), _ \
                        = rnn.bidirectional_dynamic_rnn(fw, bw, rnn_inputs, seq_len,
                                                        dtype=tf.float32, scope=scope.name + "BiRNN%d" % i)
                    rnn_inputs = tf.concat((output_fw, output_bw), 2)

                outputs = rnn_inputs

                # flatten to (N * T, F) for matrix multiplication. This will be reversed later
                outputs = tf.reshape(outputs, [-1, outputs.shape.as_list()[2]])

                W = tf.get_variable('W', initializer=tf.truncated_normal([num_hidden * 2, num_classes], stddev=0.1))
                b = tf.get_variable('B', initializer=tf.constant(0., shape=[num_classes]))

                logits = tf.matmul(outputs, W) + b

                # reshape back
                logits = tf.reshape(logits, [batch_size, -1, num_classes])

                loss = ctc_ops.ctc_loss(targets, logits, seq_len, time_major=False)

                cost = tf.reduce_mean(loss, name='cost')
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, name='optimizer')
                decoded, log_prob = ctc_ops.ctc_greedy_decoder(logits, seq_len)

                ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32, name='decoded'), targets), name='ler')

                logits = tf.nn.softmax(logits, -1, "softmax")

                # Initializate the weights and biases
                init_op = tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer())
                session.run(init_op)

                return Model(graph, session, inputs, seq_len, targets, optimizer, cost, ler, decoded, logits)

    def __init__(self, graph, session, inputs, seq_len, targets, optimizer, cost, ler, decoded, logits):
        self.graph = graph
        self.session = session
        self.inputs = inputs
        self.seq_len = seq_len
        self.targets = targets
        self.optimizer = optimizer
        self.cost = cost
        self.ler = ler
        self.decoded = decoded
        self.logits = logits

    def save(self, output_file):
        with self.graph.as_default() as g:
            saver = tf.train.Saver()
            saver.save(self.session, output_file)

    def to_sparse_matrix(self, y, y_len=None):
        batch_size = len(y)
        if y_len is not None:
            assert(batch_size == len(y_len))

            # transform [[1, 2, 5, 2], [4, 2, 1, 6, 7]]
            # to [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
            #    [1, 2, 5, 2, 4, 2, 1, 6, 7]
            #    [2, max(4, 5)]
            indices = np.concatenate([np.concatenate(
                [
                    np.full((y_len[i], 1), i),
                    np.reshape(range(y_len[i]), (-1, 1))
                ], 1) for i in range(self.batch_size)], 0)
            values = np.concatenate([
                y[i, :y_len[i]] - 1 # tensorflow ctc expects label [-1] to be blank, not 0 as ocropy
                for i in range(self.batch_size)
            ], 0)
            dense_shape = np.asarray([self.batch_size, max(y_len)])

            #print(indices, values, dense_shape)

        else:
            indices = np.concatenate([np.concatenate(
                [
                    np.full((len(y[i]), 1), i),
                    np.reshape(range(len(y[i])), (-1, 1))
                ], 1) for i in range(batch_size)], 0)
            values = np.concatenate(y, 0) - 1  # correct ctc label
            dense_shape = np.asarray([batch_size, max([len(yi) for yi in y])])
            assert(len(indices) == len(values))

        return indices, values, dense_shape

    def sparse_data_to_dense(self, x):
        batch_size = len(x)
        len_x = [xb.shape[0] for xb in x]
        max_line_length = max(len_x)

        # transform into batch (batch size, T, height)
        full_x = np.zeros((batch_size, max_line_length, x[0].shape[1]))
        for batch, xb in enumerate(x):
            full_x[batch, :len(xb)] = xb

        return full_x, len_x

    def train_sequence(self, x, y):
        # x = np.expand_dims(x, axis=0)
        # y = self.to_sparse_matrix(np.expand_dims(y, axis=0), [len(y)])
        x, len_x = self.sparse_data_to_dense(x)
        y = self.to_sparse_matrix(y)

        # with self.graph.as_default():
        cost, optimizer, logits = self.session.run([self.cost, self.optimizer, self.logits],
                                                   feed_dict={self.inputs: x,
                                                              self.seq_len: len_x,
                                                              self.targets: y,
                                                              })
        logits = np.roll(logits, 1, axis=2)
        return cost, logits

    def predict_sequence(self, x):
        x, len_x = self.sparse_data_to_dense(x)
        logits, = self.session.run([self.logits], feed_dict={self.inputs: x, self.seq_len: len_x})
        logits = np.roll(logits, 1, axis=2)
        return logits

