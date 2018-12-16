import tensorflow as tf
from utils import *

class DMNCell:
    def __init__(self, eos_vector, vocab_size, h_size, similarity_layer_size, num_passes, learning_rate, reg):
        self.eos_vector = eos_vector
        self.h_size = h_size
        # Size of the hidden layer in the similarity function G (eq 6)
        self.similarity_layer_size = similarity_layer_size
        self.vocab_size = vocab_size
        self.num_passes = num_passes
        self.memory_gru = tf.contrib.rnn.GRUCell(self.h_size)
        self.learning_rate = learning_rate
        self.reg = reg

    def run(self, input, question, answer, optimize):
        self.batch_size = tf.shape(input)[0]
        input_states, question_state = self.first_call(input, question)

        task_cond = lambda m, i: tf.less(i, self.num_passes)
        task_loop = lambda m, i: (self.memory_call(m, input_states, question_state), (i + 1))

        # The initial memory state is the question state
        last_memory_state, _ = tf.while_loop(task_cond, task_loop, [question_state, tf.constant(0)])
        output = self.last_call(question_state, last_memory_state)

        minimize_op, loss = self.optimizer(answer, output)
        accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(answer, axis=1), tf.argmax(output, axis=1))))

        minimize_op = tf.cond(optimize, lambda: minimize_op, lambda: False)
        with tf.control_dependencies([minimize_op]):
            loss = tf.identity(loss)

        return loss, accuracy, output

    def first_call(self, input, question):
        input_gru = tf.contrib.rnn.GRUCell(self.h_size)
        input_states, _ = tf.nn.dynamic_rnn(input_gru, input, dtype=tf.float32, scope='GRU_i')

        # We obtain only the hidden states at the end of the sentences
        mask = tf.reduce_all(tf.equal(input, self.eos_vector), axis=2)
        max_length = tf.reduce_sum(tf.to_float(mask), axis=1)
        self.seq_length = tf.to_int32(tf.reduce_max(max_length))

        def get_eos_states(input_state):
            i = tf.where(tf.reduce_all(tf.equal(input_states, input_state), axis=1))[0][0]
            eos_states = tf.boolean_mask(input_state, mask[i])
            padding = [[0, self.seq_length - tf.shape(eos_states)[0],], [0, 0,]]
            eos_states = tf.pad(eos_states, padding, 'constant')
            return eos_states

        ixs = tf.range(self.batch_size)
        eos_input_states = tf.map_fn(get_eos_states, input_states, dtype=tf.float32, parallel_iterations=1)

        question_gru = tf.contrib.rnn.GRUCell(self.h_size)
        _, question_state = tf.nn.dynamic_rnn(question_gru, question, dtype=tf.float32, scope='GRU_q')

        return eos_input_states, question_state

    def memory_call(self, memory_state, input_states, question_state):
        W = tf.get_variable('W_b', (self.h_size, self.h_size))

        def similarity(c):
            m, q = memory_state, question_state
            cW = tf.matmul(c, W)
            cWq = tf.reduce_sum(cW * q, axis=1, keepdims=True)
            cWm = tf.reduce_sum(cW * m, axis=1, keepdims=True)
            padding = [[0, 0,], [0, self.h_size - 1]]
            cWq = tf.pad(cWq, padding, 'constant')
            cWm = tf.pad(cWm, padding, 'constant')

            z = [c, m, q, c * m, c * q, tf.abs(c - m), tf.abs(c - q), cWq, cWm]
            z = tf.reshape(tf.stack(z, axis=1), (self.batch_size, len(z) * self.h_size))
            h_layer = tf.layers.dense(z, self.similarity_layer_size, activation=tf.nn.tanh)
            gate = tf.layers.dense(h_layer, 1, activation=tf.nn.sigmoid)
            return gate

        swapped_input = tf.transpose(input_states, [1, 0, 2])
        gates = tf.map_fn(similarity, swapped_input)
        gates = tf.transpose(gates, [1, 0, 2])
        episode_gru = tf.contrib.rnn.GRUCell(self.h_size)

        episode_cond = lambda state, i: tf.less(i, self.seq_length)
        def episode_loop(state, i):
            next_state = episode_gru(input_states[:, i], state)[0]
            next_state = gates[:, i] * next_state + (1 - gates[:, i]) * state
            return next_state, (i + 1)

        initial_state = episode_gru.zero_state(self.batch_size, dtype=tf.float32)
        i = tf.constant(0)
        episode, _ = tf.while_loop(episode_cond, episode_loop, [initial_state, i])
        next_memory_state = self.memory_gru(episode, memory_state)[0]

        return next_memory_state

    def last_call(self, question_states, last_memory_state):
        answer_gru = tf.contrib.rnn.GRUCell(self.h_size)
        answer_state = answer_gru(question_states, last_memory_state)[0]
        output = tf.layers.dense(answer_state, self.vocab_size, use_bias=False)
        return output

    def optimizer(self, answer, output):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                            if 'bias' not in v.name]) * self.reg
        loss = tf.losses.softmax_cross_entropy(answer, output) + l2_loss
        minimize = optimizer.minimize(loss)
        return minimize, loss

'''
Next steps
* make jupybooks
* check airplane

input_states = tf.transpose(input_states, [1, 0, 2]) #(max_time, batch_size, input_depth
max_time, batch_size, input_depth = tf.shape(input_states)
sequence_length = tf.Variable(max_time, shape=(batch_size,), dtype=tf.int32)
input_states_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
input_states_ta = inputs_states_ta.unstack(input_states)

note: optimizer should be in model.py
think: is the focus on the last words (given by the rnn) good?
why are we using the same weight matrix in similarity measure?
google would be better i think if it redirects you to the first page and has a small square in the bottom right with the other results.
how do you say something that is valid in all the meta versions? For instance, I can say: be consistant in life. then, that rule applies to itself. So you need to be consistant in being consistant. But, if I limit myself to some scope (eg be consistant in programming) then I'm not talking about the consistantcy about following the rule (which is interesting, because not being consistant for the rule produces a similar effect to not being consistant for programming)
what if every neuron in a neural net is a GRUCell? It's prohibitely expensive, but who cares? How would be that structure?
'''
