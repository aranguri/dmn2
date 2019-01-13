import tensorflow as tf
from utils import *
# from tensorflow.contrib.cudnn_rnn import CudnnCompatibleGRUCell as GRU #GPU version
from tensorflow.contrib.rnn import GRUCell as GRU #CPU version

class DMNCell:
    def __init__(self, eos_vector, vocab_size, h_size, similarity_layer_size, learning_rate, optimize):
        self.eos_vector = eos_vector
        self.vocab_size = vocab_size
        self.h_size = h_size
        self.similarity_layer_size = similarity_layer_size
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.optimize = optimize

    def run(self, input, question, supporting_hot):
        # Running
        self.batch_size = tf.shape(input)[0]
        input_states, question_state = self.first_call(input, question)
        gates = self.memory_call(input_states, question_state)

        # Minimizing loss
        supporting = tf.one_hot(supporting_hot, self.seq_length)
        loss = tf.losses.softmax_cross_entropy(supporting, gates)
        def minimize_fn():
            minimize = self.minimize_op(loss)
            with tf.control_dependencies([minimize]):
                return tf.identity(loss)
        maybe_minimize_op = tf.cond(self.optimize, minimize_fn, lambda: tf.identity(loss))

        with tf.control_dependencies([maybe_minimize_op]):
            loss = tf.identity(loss)

        # Accuracy
        gates_hot = tf.argmax(gates, axis=2)
        accuracy = tf.reduce_mean(tf.to_float(tf.equal(supporting_hot, tf.to_int32(gates_hot))))

        return loss, accuracy, gates

    def first_call(self, input, question):
        input_gru = GRU(self.h_size)
        input_states, _ = tf.nn.dynamic_rnn(input_gru, input, dtype=tf.float32, scope='GRU_i')

        #alternative: do this by using tf.where(mask, input, tf.zeros_like(input)) and then (somehow) removing the zeros
        mask = tf.reduce_all(tf.equal(input, self.eos_vector), axis=2)
        max_length = tf.reduce_sum(tf.to_float(mask), axis=1)
        self.seq_length = tf.to_int32(tf.reduce_max(max_length))

        # This operation doesn't seem to be GPU-friendly
        def get_eos_states(i):
            # rolled_mask = tf.roll(mask[i], -1, 0)
            # We shift the mask one to the left to avoid selecting the state where we input the dot
            eos_states = tf.boolean_mask(input_states[i], mask[i])
            padding = [[0, self.seq_length - tf.shape(eos_states)[0],], [0, 0,]]
            return tf.pad(eos_states, padding, 'constant')

        eos_input_states = tf.map_fn(get_eos_states, tf.range(self.batch_size), tf.float32)

        question_gru = GRU(self.h_size)
        _, question_state = tf.nn.dynamic_rnn(question_gru, question, dtype=tf.float32, scope='GRU_q')

        return eos_input_states, question_state

    def memory_call(self, input_states, question_state):
        # with ts(question_state):
        question_tiled = tf.tile(question_state, [self.seq_length, 1])
        question_stacked = tf.reshape(question_tiled, (self.batch_size, self.seq_length, self.h_size))
        input = tf.concat((input_states, question_stacked), axis=2)
        h1 = tf.layers.dense(input, self.similarity_layer_size, activation=tf.nn.tanh)
        # h2 = tf.layers.dense(h1, self.similarity_layer_size, activation=tf.nn.tanh)
        out = tf.layers.dense(h1, 1)
        gates = tf.transpose(out, [0, 2, 1])

        return gates

    def minimize_op(self, loss):
        return self.optimizer.minimize(loss)

#64x10x512
#64x512
