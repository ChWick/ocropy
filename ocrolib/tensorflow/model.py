import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, DropoutWrapper
from tensorflow.python.ops import ctc_ops
import numpy as np
import re


class Model:
    @staticmethod
    def default_model_settings():
        return {
            "conv_pool": [
                {
                    "filters": 40,
                    "kernel_size": [3, 3],
                    "pool_size": [2, 2],
                },
                {
                    "filters": 60,
                    "kernel_size": [3, 3],
                    "pool_size": [2, 2],
                },
            ],
            "lstm": [
                100
            ],
            "use_peepholes": False,
            "ctc_merge_repeated": False,
        }

    @staticmethod
    def write_model_settings(model_settings):
        out_parts = []
        for layer in model_settings["layers"]:
            if layer["type"] == "cnn":
                out_parts.append("cnn_%d_%dx%d" % (layer["filters"], layer["kernel_size"][0], layer["kernel_size"][1]))
            elif layer["type"] == "pool":
                out_parts.append("pool_%dx%d" % (layer["kernel_size"][0], layer["kernel_size"][1]))
            elif layer["type"] == "lstm":
                out_parts.append("lstm_%d" % (layer["hidden"], ))
            else:
                raise Exception("Unknown layer type '%s'" % layer["type"])

        for flag in ["use_peepholes", "ctc_merge_repeated"]:
            out_parts.append("%s_%s" % (flag, ("false", "true")[model_settings[flag]]))

        return "__".join(out_parts)



    @staticmethod
    def parse_model_settings(str):
        cnn_matcher = re.compile("^([\d]+)(:([\d]+)(x([\d]+))?)?$")
        pool_matcher = re.compile("^([\d]+)(x([\d]+))?$")
        params = str.split(",")
        model = []
        lstm_appeared = False
        params_dict = {
            "ctc_merge_repeated": True,
            "use_peepholes": False,
            "layers": model,
        }
        for param in params:
            label, value = tuple(param.split("="))
            if label == "ctc_merge_repeated":
                params_dict["ctc_merge_repeated"] = value.lower == "true"
            elif label == "use_peepholes":
                params_dict["use_peepholes"] = value.lower == "true"
            elif label == "lstm":
                lstm = {
                    "type": "lstm",
                    "hidden": int(value),
                }
                model.append(lstm)
            elif label == "cnn":
                if lstm_appeared:
                    raise Exception("LSTM layers must be placed proceeding to CNN/Pool")

                match = cnn_matcher.match(value)
                if match is None:
                    raise Exception("CNN structure needs: cnn=[filters]:[h]x[w]")

                match = match.groups()
                kernel_size = [2, 2]
                if match[1] is not None:
                    kernel_size = [int(match[2])] * 2
                if match[3] is not None:
                    kernel_size = [int(match[2]), int(match[4])]

                cnn = {
                    "type": "cnn",
                    "filters": int(match[0]),
                    "kernel_size": kernel_size,
                }
                model.append(cnn)
            elif label == "pool":
                if lstm_appeared:
                    raise Exception("LSTM layers must be placed proceeding to CNN/Pool")
                match = pool_matcher.match(value)
                if match is None:
                    raise Exception("Pool structure needs: pool=[h];[w]")

                match = match.groups()
                kernel_size = [int(match[0])] * 2
                if match[1] is not None:
                    kernel_size = [int(match[0]), int(match[2])]

                pool = {
                    "type": "pool",
                    "kernel_size": kernel_size
                }
                model.append(pool)

        return params_dict



    @staticmethod
    def load(filename, threads):
        print("Loading tensorflow model from root %s" % filename)
        graph = tf.Graph()
        with graph.as_default() as g:
            session = tf.Session(graph=graph,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=threads,
                                                       inter_op_parallelism_threads=threads),
                                )
            with tf.variable_scope("", reuse=False) as scope:

                saver = tf.train.import_meta_graph(filename + '.meta')
                saver.restore(session, filename)

                l_rate = g.get_tensor_by_name("l_rate:0")
                inputs = g.get_tensor_by_name("inputs:0")
                seq_len = g.get_tensor_by_name("seq_len:0")
                try:
                    seq_len_out = g.get_tensor_by_name("seq_len_out:0")
                except:
                    print("loaded old model!")
                    seq_len_out = seq_len / 4

                targets = tf.SparseTensor(
                    g.get_tensor_by_name("targets/indices:0"),
                    g.get_tensor_by_name("targets/values:0"),
                    g.get_tensor_by_name("targets/shape:0"))
                cost = g.get_tensor_by_name("cost:0")
                train_op = g.get_operation_by_name('train_op')
                ler = g.get_tensor_by_name("ler:0")
                decoded = (
                    g.get_tensor_by_name("decoded_indices:0"),
                    g.get_tensor_by_name("decoded_values:0"),
                    g.get_tensor_by_name("decoded_shape:0")
                    )
                logits = g.get_tensor_by_name("softmax:0")

                return Model(graph, session, inputs, seq_len, seq_len_out, targets, train_op, cost, ler, decoded, logits, l_rate)

    @staticmethod
    def create(num_features, num_classes, model_settings, reuse_variables=False, threads=1):
        print("Creating tf graph with settings: %s" % model_settings)
        graph = tf.Graph()
        with graph.as_default():
            session = tf.Session(graph=graph,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=threads,
                                                       inter_op_parallelism_threads=threads,
                                                       ))

            inputs = tf.placeholder(tf.float32, shape=(None, None, num_features), name="inputs")
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.placeholder(tf.int32, shape=(None,), name="seq_len")
            targets = tf.sparse_placeholder(tf.int32, shape=(None, None), name="targets")
            l_rate = tf.placeholder(tf.float32, shape=(), name="l_rate")

            with tf.variable_scope("", reuse=reuse_variables) as scope:
                has_conv_or_pool = model_settings["layers"][0]["type"] != "lstm"

                if has_conv_or_pool:
                    cnn_inputs = tf.reshape(inputs, [batch_size, -1, num_features, 1])
                    shape = seq_len, num_features

                    layers = [cnn_inputs]
                    last_num_filters = 1

                    for model in [l for l in model_settings["layers"] if l["type"] != "lstm"]:
                        if model["type"] == "cnn":
                            layers.append(tf.layers.conv2d(
                                inputs=layers[-1],
                                filters=model["filters"],
                                kernel_size=model["kernel_size"],
                                padding="same",
                                activation=tf.nn.relu,
                            ))
                            last_num_filters = model["filters"]
                        elif model["type"] == "pool":
                            layers.append(tf.layers.max_pooling2d(
                                inputs=layers[-1],
                                pool_size=model["kernel_size"],
                                strides=model["kernel_size"],
                            ))

                            shape = (tf.to_int32(shape[0] / model["kernel_size"][0]),
                                     shape[1] / model["kernel_size"][1])
                        else:
                            raise Exception("Unknown layer of type %s" % model["type"])

                    lstm_seq_len, lstm_num_features = shape
                    rnn_inputs = tf.reshape(layers[-1],
                                            [batch_size, tf.shape(layers[-1])[1],
                                             last_num_filters * lstm_num_features])

                else:
                    rnn_inputs = inputs
                    lstm_seq_len = seq_len

                lstm_layers = [l for l in model_settings["layers"] if l['type'] == "lstm"]

                if len(lstm_layers) > 0:
                    def get_lstm_cell(num_hidden, use_peepholse=model_settings["use_peepholes"]):
                        return LSTMCell(num_hidden,
                                        use_peepholes=use_peepholse,
                                        reuse=reuse_variables,
                                        initializer=tf.initializers.random_uniform(-0.1, 0.1),
                                        #cell_clip=20,
                                        #proj_clip=20,
                                        #activation=tf.sigmoid,
                                        )

                    for i, lstm in enumerate(lstm_layers):
                        fw, bw = get_lstm_cell(lstm["hidden"]), get_lstm_cell(lstm["hidden"])
                        (output_fw, output_bw), _ \
                            = rnn.bidirectional_dynamic_rnn(fw, bw, rnn_inputs, lstm_seq_len,
                                                            dtype=tf.float32, scope=scope.name + "BiRNN%d" % i)
                        rnn_inputs = tf.concat((output_fw, output_bw), 2)

                    output_size = lstm_layers[-1]["hidden"] * 2
                else:
                    raise Exception("Currently unsupported: no lstm layer")

                outputs = rnn_inputs

                # flatten to (N * T, F) for matrix multiplication. This will be reversed later
                outputs = tf.reshape(outputs, [-1, outputs.shape.as_list()[2]])

                W = tf.get_variable('W', initializer=tf.random_uniform([output_size, num_classes], -0.1, 0.1))
                b = tf.get_variable('B', initializer=tf.constant(0., shape=[num_classes]))

                logits = tf.matmul(outputs, W) + b

                # reshape back
                logits = tf.reshape(logits, [batch_size, -1, num_classes])

                softmax = tf.nn.softmax(logits, -1, "softmax")


                # time major (required for ctc decoder)
                time_major_logits = tf.transpose(logits, (1, 0, 2), name='time_major_logits')

                # ctc predictions
                loss = ctc_ops.ctc_loss(targets,
                                        time_major_logits,
                                        lstm_seq_len,
                                        time_major=True,
                                        ctc_merge_repeated=model_settings["ctc_merge_repeated"],
                                        ignore_longer_outputs_than_inputs=True)
                decoded, log_prob = ctc_ops.ctc_greedy_decoder(time_major_logits, lstm_seq_len, merge_repeated=model_settings["ctc_merge_repeated"])
                #decoded, log_prob = ctc_ops.ctc_beam_search_decoder(time_major_logits, lstm_seq_len, merge_repeated=model_settings["merge_repeated"])
                decoded = decoded[0]
                sparse_decoded = (
                    tf.identity(decoded.indices, name="decoded_indices"),
                    tf.identity(decoded.values, name="decoded_values"),
                    tf.identity(decoded.dense_shape, name="decoded_shape"),
                )



                cost = tf.reduce_mean(loss, name='cost')
                optimizer = tf.train.AdamOptimizer(1e-3)
                #optimizer = tf.train.MomentumOptimizer(1e-3, 0.9)
                # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, name='optimizer')
                gvs = optimizer.compute_gradients(cost)
                capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
                train_op = optimizer.apply_gradients(gvs, name='train_op')

                ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded, tf.int32), targets), name='ler')


                # Initializate the weights and biases
                init_op = tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer())
                session.run(init_op)

                lstm_seq_len = tf.identity(lstm_seq_len, "seq_len_out")

                return Model(graph, session, inputs, seq_len, lstm_seq_len, targets, train_op, cost, ler, sparse_decoded, softmax, l_rate)

    def __init__(self, graph, session, inputs, seq_len, seq_len_out, targets, optimizer, cost, ler, sparse_decoded, logits, l_rate):
        self.graph = graph
        self.session = session
        self.inputs = inputs
        self.seq_len = seq_len
        self.seq_len_out = seq_len_out
        self.targets = targets
        self.optimizer = optimizer
        self.cost = cost
        self.ler = ler
        self.decoded = sparse_decoded
        self.logits = logits
        self.l_rate = l_rate
        self.default_learning_rate = -1

    def load_weights(self, model_file):
        with self.graph.as_default() as g:
            all_var_names = [v for v in tf.global_variables() if not v.name.startswith("W") and not v.name.startswith("B")]
            print(all_var_names)
            saver = tf.train.Saver(all_var_names)

            # Restore variables from disk.
            saver.restore(self.session, model_file)
            print("Model restored")

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

        # return full_x, len_x
        return full_x, [l for l in len_x]

    @staticmethod
    def sparse_to_lists(sparse, shift_values = 0):
        #indices, values, dense_shape = sparse.indices, sparse.values, sparse.dense_shape
        indices, values, dense_shape = sparse

        out = [[] for _ in range(dense_shape[0])]

        for index, value in zip(indices, values):
            x, y = tuple(index)
            out[x].append(value + shift_values)

        return out

    def train_sequence(self, x, y):
        #print(y[0].shape, x[0].shape, self.default_learning_rate)
        # x = np.expand_dims(x, axis=0)
        # y = self.to_sparse_matrix(np.expand_dims(y, axis=0), [len(y)])
        x, len_x = self.sparse_data_to_dense(x)
        y = self.to_sparse_matrix(y)


        # with self.graph.as_default():
        cost, optimizer, logits, ler, decoded = self.session.run([self.cost, self.optimizer, self.logits, self.ler, self.decoded],
                                                   feed_dict={self.inputs: x,
                                                              self.seq_len: len_x,
                                                              self.targets: y,
                                                              self.l_rate: self.default_learning_rate,
                                                              })
        logits = np.roll(logits, 1, axis=2)
        return cost, logits, ler, Model.sparse_to_lists(decoded, shift_values=1)

    def predict_sequence(self, x):
        x, len_x = self.sparse_data_to_dense(x)
        logits, seq_len_out = self.session.run([self.logits, self.seq_len_out], feed_dict={self.inputs: x, self.seq_len: len_x})
        logits = np.roll(logits, 1, axis=2)
        return logits, seq_len_out

    def decode_sequence(self, x):
        x, len_x = self.sparse_data_to_dense(x)
        logits, seq_len, decoded, = self.session.run([self.logits, self.seq_len_out, self.decoded], feed_dict={self.inputs: x, self.seq_len: len_x})
        logits = np.roll(logits, 1, axis=2)
        decoded = Model.sparse_to_lists(decoded, shift_values=1)
        return logits, seq_len, decoded

