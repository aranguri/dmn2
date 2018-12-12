import tensorflow as tf
from utils import *

class DMNCell:
    def __init__(self, eos_vector, input_h_size, question_h_size, episode_h_size, similarity_layer_size, num_passes):
        self.eos_vector = eos_vector
        self.input_h_size = input_h_size
        self.question_h_size = question_h_size
        self.episode_h_size = episode_h_size
        #Size of the hidden layer in the similarity function G (eq 6)
        self.similarity_layer_size = similarity_layer_size
        self.num_passes = num_passes
        # We use h_size = question_h_size, for memory is initialized with the question
        self.memory_gru = tf.contrib.rnn.GRUCell(self.question_h_size)

    def run(self, input_, question):
        input_states, question_states = self.first_call(input_, question)

        def task_cond(memory_state, i):
            return tf.less(i, self.num_passes)

        def task_loop(memory_state, i):
            memory_state = self.memory_call(memory_state, input_states, question_states)
            return memory_state, i

        memory_state = question #CHECK: are we passing values by value and not by reference?
        i = tf.constant(0)
        final_memory_state = tf.while_loop(task_cond, task_loop, [memory_state, i])
        y = self.last_call(final_memory_state)
        return y

    def first_call(self, input_, question):
        input_gru = tf.contrib.rnn.GRUCell(self.input_h_size)
        # with tf.control_dependencies([tp(input_ == self.eos_vector)]):
        input_states, _ = tf.nn.dynamic_rnn(input_gru, input_, dtype=tf.float32)
        self.input_states = tf.boolean_mask(input_states, tf.equal(input_, self.eos_vector))

        question_gru = tf.contrib.rnn.GRUCell(self.question_h_size)
        _, self.question_state = tf.nn.dynamic_rnn(question_gru, question, dtype=tf.float32)

        seq_length = tf.shape(input_states)[1] #CHECK: this number can be wrong

    def memory_call(self, memory_state, input_states, question_states):
        gate = tf.map_fn(similarity, input_states)
        episode_gru = tf.contrib.rnn.GRUCell(self.episode_h_size)

        def similarity(c):
            m, q = memory_state, question_state
            cW = tf.matmul(tf.transpose(c), W)
            cWq = tf.matmul(cW, q)
            cWm = tf.matmul(cW, m)
            # we are gonna need to pad cWq and ...
            z = tf.stack((c, m, q, c * m, c * q, tf.abs(c - m), tf.abs(c - q), cWq, cWm))
            layer1 = tf.layers.dense(z, similarity_layer_size, activation=tf.nn.tanh)
            out = tf.layers.dense(layer1, 1, activation=tf.nn.sigmoid)
            return out

        def episode_cond(state, i):
            return tf.less(i, seq_length)

        def episode_loop(state, i):
            input_ = input_states[i] #CHECK: if it doesn't work, use TArray
            _, next_state = episode_gru(input_, state)
            next_state = gate[i] * next_state + (1 - gate[i]) * state
            return next_state, (i + 1)

        i = tf.constant(0)
        initial_state = episode_gru.zero_state(self.batch_size, dtype=tf.float32)
        episode = tf.while_loop(episode_cond, episode_loop, [initial_state, i])

        next_memory_state, _ = self.memory_gru(episode, memory_state)
        return next_memory_state

    def last_call(self, final_memory_state):
        answer_h_size = len(final_memory_state)
        answer_gru = tf.contrib.rnn.GRUCell(answer_h_size)
        _, answer_state = answer_gru(question, final_memory_state)
        return tf.nn.softmax(tf.layers.dense(answer_state, vocab_size, use_bias=False), axis=1) #CHECK: axis, and it could be changed to activations=tf.nn.softmargax


'''
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

def episode_loop(time, cell_output, cell_state, loop_state):
    if cell_output is None:
        next_cell_state = cell.zero_state(batch_size, tf.float32)
    else:
        next_cell_state = g * cell_state + (1 - g) * prev_cell_state

    elements_finished = (time >= sequence_length)
    finished = tf.reduce_all(elements_finished)
    next_input = tf.cond(
        finished,
        lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
        lambda: input_states_ta.read(time))

    return (finished, next_input, next_cell_state,
        None, next_loop_state)


def loop_fn(time, cell_output, cell_state, loop_state):
    if cell_output is None:  # time == 0
        next_cell_state = cell.zero_state(batch_size, tf.float32)
    else:
        next_cell_state = cell_state

    next_loop_state = None
    return (elements_finished, next_input, next_cell_state,
          None, next_loop_state)

_, final_state, _ = raw_rnn(episode_gru, loop_fn)
'''
