import tensorflow as tf
from utils import *
from tensorflow.contrib.cudnn_rnn import CudnnCompatibleGRUCell as GRU #GPU version
# from tensorflow.contrib.rnn import GRUCell as GRU #CPU version

class DMNCell:
    def __init__(self, eos_vector, vocab_size, h_size, similarity_layer_size, output_hidden_size, learning_rate, alpha, beta, gates_acc_threshold):
        self.eos_vector = eos_vector
        self.vocab_size = vocab_size
        self.h_size = h_size
        self.similarity_layer_size = similarity_layer_size
        self.output_hidden_size = output_hidden_size
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.alpha = alpha
        self.beta = beta
        self.gates_acc_threshold = gates_acc_threshold

    def run(self, input, question, answer_hot, supporting):
        # Running
        self.batch_size = tf.shape(input)[0]
        input_states, question_state = self.first_call(input, question)
        gates_hot = self.get_gates(input_states, question_state, question_state)
        episode = self.get_episode(input_states, gates_hot)
        memory = self.get_memory(question_state, episode)
        output_hot = self.get_output(question_state, memory)

        # Loss and accuracy
        supporting_hot = tf.one_hot(supporting, self.seq_length)
        answer = tf.argmax(answer_hot, axis=1)
        output = tf.argmax(output_hot, axis=1)
        gates = tf.argmax(gates_hot, axis=2, output_type=tf.int32)
        output_acc = tf.reduce_mean(tf.to_float(tf.equal(output, answer)))
        gates_acc = tf.reduce_mean(tf.to_float(tf.equal(gates, supporting)))

        self.alpha = tf.cond(gates_acc > self.gates_acc_threshold, lambda: 1., lambda: 0.)
        loss, output_loss, gates_loss = self.get_loss(output_hot, gates_hot, answer_hot, supporting_hot)

        return loss, (output_loss, gates_loss, output_acc, gates_acc)

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

    def get_gates(self, c, q, m):
        q = tf.expand_dims(q, 1)
        m = tf.expand_dims(m, 1)
        qs = tf.tile(q, [1, self.seq_length, 1])
        ms = tf.tile(m, [1, self.seq_length, 1])
        W = tf.get_variable('W', (self.h_size, self.h_size))
        cW = tf.einsum('ijk,kl->ijl', c, W)
        cWq = tf.einsum('ijk,ilk->ij', cW, q)
        cWq = tf.expand_dims(cWq, 2)
        cWm = tf.einsum('ijk,ilk->ij', cW, m)
        cWm = tf.expand_dims(cWm, 2)

        z = (c, qs, ms, tf.abs(c - q), tf.abs(c - m), c * q, c * m, cWq, cWm)
        z_concat = tf.concat(z, axis=2)
        input = tf.reshape(z_concat, (self.batch_size, self.seq_length, (len(z) - 2) * self.h_size + 2))
        h1 = tf.layers.dense(input, self.similarity_layer_size, activation=tf.nn.tanh)
        out = tf.layers.dense(h1, 1)#, activation=tf.nn.sigmoid)
        gates = tf.transpose(out, [0, 2, 1])

        return gates

    def get_episode(self, input_states, gates):
        episode_gru = GRU(self.h_size)
        episode_cond = lambda state, i: tf.less(i, self.seq_length)

        def episode_loop(state, i):
            next_state = episode_gru(input_states[:, i], state)[0]
            next_state = gates[:, :, i] * next_state + (1 - gates[:, :, i]) * state
            return next_state, (i + 1)

        initial_state = episode_gru.zero_state(self.batch_size, dtype=tf.float32)
        i = tf.constant(0)
        episode, _ = tf.while_loop(episode_cond, episode_loop, [initial_state, i])
        return episode

    def get_memory(self, memory_state, episode):
        memory_gru = GRU(self.h_size)
        next_memory_state = memory_gru(episode, memory_state)[0]
        return next_memory_state

    def get_output(self, question_state, memory):
        input = tf.concat((question_state, memory), axis=1)
        hidden = tf.layers.dense(input, self.output_hidden_size, activation=tf.nn.relu)
        output = tf.layers.dense(hidden, self.vocab_size)
        return output

    def get_loss(self, output, gates, answer, supporting):
        output_loss = tf.losses.softmax_cross_entropy(answer, output)
        gates_loss = tf.losses.softmax_cross_entropy(supporting, gates)
        loss = self.alpha * output_loss + self.beta * gates_loss
        return loss, output_loss, gates_loss

    def minimize_op(self, loss):
        with tp(tf.constant(0)):
            loss = tf.identity(loss)
        return self.optimizer.minimize(loss)
