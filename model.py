import tensorflow as tf
from utils import *

class DMNCell:
    def __init__(self, eos_vector, vocab_size, h_size, similarity_layer_size, num_passes, learning_rate, reg):
        self.eos_vector = eos_vector
        self.h_size = h_size
        # similarity_layer_size is the size of the hidden layer in the similarity function G (eq 6)
        self.similarity_layer_size = similarity_layer_size
        self.vocab_size = vocab_size
        self.num_passes = num_passes
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.reg = reg
        # Params for prioritizing one loss over the other
        self.alpha = 0
        self.beta = 1
        self.memory_gru = tf.contrib.rnn.GRUCell(self.h_size)

    def run(self, input, question, answer, supporting, optimize):
        self.batch_size = tf.shape(input)[0]
        # Running
        input_states, question_state = self.first_call(input, question)

        task_cond = lambda m, g, i: tf.less(i, self.num_passes)
        def task_loop(memory, gate, i):
            next_memory, next_gate = self.memory_call(memory, input_states, question_state)
            gate = tf.concat((gate, next_gate), axis=1)
            return next_memory, gate, (i + 1)
        gate = tf.fill((self.batch_size, 0, self.seq_length), 0.0)
        ixs = tf.constant(0.0)
        shapes = [question_state.get_shape(), tf.TensorShape((None, None, None)), ixs.get_shape()]
        last_memory_state, gates, _ = tf.while_loop(task_cond, task_loop, [question_state, gate, ixs], shapes)

        output = self.last_call(question_state, last_memory_state)

        # optimizing
        supporting = tf.one_hot(supporting, self.seq_length)

        loss = self.get_loss(answer, output, supporting, gates)
        accuracy = tf.reduce_mean(tf.to_float(tf.argmax(answer, axis=1) == tf.argmax(output, axis=1)))

        # We execute the minimize_op iff optimize == True
        def minimize_fn():
            minimize = self.minimize_op(loss)
            with tf.control_dependencies([minimize]):
                return tf.identity(loss)

        maybe_minimize_op = tf.cond(optimize, minimize_fn, lambda: tf.identity(loss))

        with tf.control_dependencies([maybe_minimize_op]):
            loss = tf.identity(loss)

        return loss, accuracy, output, gates

    def first_call(self, input, question):
        input_gru = tf.contrib.rnn.GRUCell(self.h_size)
        input_states, _ = tf.nn.dynamic_rnn(input_gru, input, dtype=tf.float32, scope='GRU_i')

        # We obtain only the hidden states at the end of the sentences
        mask = tf.reduce_all(tf.equal(input, self.eos_vector), axis=2)
        max_length = tf.reduce_sum(tf.to_float(mask), axis=1)
        self.seq_length = tf.to_int32(tf.reduce_max(max_length))

        def get_eos_states(i):
            eos_states = tf.boolean_mask(input_states[i], mask[i])
            padding = [[0, self.seq_length - tf.shape(eos_states)[0],], [0, 0,]]
            return tf.pad(eos_states, padding, 'constant')

        eos_input_states = tf.map_fn(get_eos_states, tf.range(self.batch_size), tf.float32)

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
        gates = tf.transpose(gates, [0, 2, 1])

        return next_memory_state, gates

    def last_call(self, question_states, last_memory_state):
        answer_gru = tf.contrib.rnn.GRUCell(self.h_size)
        answer_state = answer_gru(question_states, last_memory_state)[0]
        output = tf.layers.dense(answer_state, self.vocab_size, use_bias=False)

        return output

    def get_loss(self, answer, output, supporting, gates):
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                            if 'bias' not in v.name]) * self.reg

        gates_loss = tf.losses.softmax_cross_entropy(supporting, gates)
        output_loss = tf.losses.softmax_cross_entropy(answer, output)
        return self.beta * output_loss + self.alpha * gates_loss# + l2_loss

    def minimize_op(self, loss):
        return self.optimizer.minimize(loss)

'''
think: is the focus on the last words (given by the rnn) good?
why are we using the same weight matrix in similarity measure?
google would be better i think if it redirects you to the first page and has a small square in the bottom right with the other results.
how do you say something that is valid in all the meta versions? For instance, I can say: be consistant in life. then, that rule applies to itself. So you need to be consistant in being consistant. But, if I limit myself to some scope (eg be consistant in programming) then I'm not talking about the consistantcy about following the rule (which is interesting, because not being consistant for the rule produces a similar effect to not being consistant for programming)
what if every neuron in a neural net is a GRUCell? It's prohibitely expensive, but who cares? How would be that structure?
This thing of talking to myself and listening to music to get more energies really help (I don't know if ¬sleeping is scalable though, but talkking to myself is) INdeed it made me more focused on what I was doing that days in the piecita/green where I was wondering in thoughts but having silence
wrt ww it seems I'm donig everythign wrong -- or worse, I'm doing nothing :)

Terms
Lazy evaluation: evaluating an expression iff it's needed
'''
