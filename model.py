import tensorflow as tf
from utils import *
from tensorflow.contrib.cudnn_rnn import CudnnCompatibleGRUCell as GRU #GPU version
# from tensorflow.contrib.rnn import GRUCell as GRU #CPU version

class DMNCell:
    def __init__(self, eos_vector, vocab_size, h_size, similarity_layer_size, learning_rate):
        self.eos_vector = eos_vector
        self.h_size = h_size
        # similarity_layer_size is the size of the hidden layer in the similarity function G (eq 6)
        self.similarity_layer_size = similarity_layer_size
        self.vocab_size = vocab_size
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.memory_gru = GRU(self.h_size)

    def run(self, input, question, supporting):
        self.batch_size = tf.shape(input)[0]
        # Running
        input_states, question_state = self.first_call(input, question)
        _, gates = self.memory_call(question_state, input_states, question_state)

        # optimizing
        supporting = tf.one_hot(supporting, self.seq_length)
        loss = tf.losses.softmax_cross_entropy(supporting, gates)
        minimize = self.optimizer.minimize(loss)
        with tf.control_dependencies([minimize]):
            accuracy = tf.constant(1.)

        return loss, accuracy, gates

    def first_call(self, input, question):
        input_gru = GRU(self.h_size)
        input_states, _ = tf.nn.dynamic_rnn(input_gru, input, dtype=tf.float32, scope='GRU_i')

        # We obtain only the hidden states at the end of the sentences
        mask = tf.reduce_all(tf.equal(input, self.eos_vector), axis=2)
        max_length = tf.reduce_sum(tf.to_float(mask), axis=1)
        self.seq_length = tf.to_int32(tf.reduce_max(max_length))

        # This operation doesn't seem to be GPU-friendly
        def get_eos_states(i):
            rolled_mask = tf.roll(mask[i], -1, 0)
            # We shift the mask one to the left to avoid selecting the state where we input the dot
            eos_states = tf.boolean_mask(input_states[i], rolled_mask)
            padding = [[0, self.seq_length - tf.shape(eos_states)[0],], [0, 0,]]
            return tf.pad(eos_states, padding, 'constant')

        eos_input_states = tf.map_fn(get_eos_states, tf.range(self.batch_size), tf.float32)

        question_gru = GRU(self.h_size)
        _, question_state = tf.nn.dynamic_rnn(question_gru, question, dtype=tf.float32, scope='GRU_q')

        return eos_input_states, question_state

    def memory_call(self, memory_state, input_states, question_state):
        W = tf.get_variable('W_b', (self.h_size, self.h_size))
        m, q = memory_state, question_state

        def similarity(c):
            cW = tf.matmul(c, W)
            cWq = tf.reduce_sum(cW * q, axis=1, keepdims=True)
            cWm = tf.reduce_sum(cW * m, axis=1, keepdims=True)
            padding = [[0, 0,], [0, self.h_size - 1]]
            cWq = tf.pad(cWq, padding, 'constant')
            cWm = tf.pad(cWm, padding, 'constant')

            z = [c, m, q, c * m, c * q, tf.abs(c - m), tf.abs(c - q), cWq, cWm]
            z_stacked = tf.reshape(tf.stack(z, axis=1), (self.batch_size, len(z) * self.h_size))
            h_layer = tf.layers.dense(z_stacked, self.similarity_layer_size, activation=tf.nn.tanh)
            gate = tf.layers.dense(h_layer, 1, activation=tf.nn.sigmoid)
            return gate

        swapped_input = tf.transpose(input_states, [1, 0, 2])
        swapped_gates = tf.map_fn(similarity, swapped_input)
        gates = tf.transpose(swapped_gates, [1, 2, 0])

        return None, gates
