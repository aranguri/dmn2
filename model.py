import tensorflow as tf

input_h_size = 32
question_h_size = 32
episode_h_size = 32

similarity_layer_size = 16 #Size of the hidden layer in the similarity function G (eq 6)

input_gru = tf.contrib.rnn.GRUCell(input_h_size)
input_states, _ = tf.nn.dynamic_rnn(input_gru, input_, dtype=tf.float32)

question_gru = tf.contrib.rnn.GRUCell(question_h_size)
_, question_state = tf.nn.dynamic_rnn(question_gru, question, dtype=tf.float32)
input_states = tf.boolean_mask(input_states, (input_ == EOS_INDEX))

seq_length = tf.shape(input_states)[1] #CHECK: this number can be wrong
loop {

__init__
self.memory_gru

gate = tf.map_fn(similarity, input_states)
episode_gru = tf.contrib.rnn.GRUCell(episode_h_size)

def similarity(c):
    m, q = memory_state, question_state
    cW = tf.matmul(tf.transpose(c), W)
    cWq = tf.matmul(cW, q)
    cWm = tf.matmul(cW, m)
    # we are gonna need to pad cWq and ...
    z = tf.stack((c, m, q, c * m, c * q, tf.abs(c - m), tf.abs(c - q), cWq, cWm)
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
initial_state = episode_gru.zero_state(batch_size, dtype=tf.float32)
episode = tf.while_loop(episode_cond, episode_loop, [initial_state, i])

memory_next_state = self.memory_gru(episode, memory_state)

}

memory_state #assume it's the final state
answer_gru = tf.contrib.rnn.GRUCell(len(memory_state))
_, answer_state = answer_gru(question, memory_state)
y = tf.nn.softmax(tf.layers.dense(answer_state, vocab_size, use_bias=False), axis=1) #CHECK: axis, and it could be changed to activations=tf.nn.softmargax






'''
input_states = tf.transpose(input_states, [1, 0, 2]) #(max_time, batch_size, input_depth
max_time, batch_size, input_depth = tf.shape(input_states)
sequence_length = tf.Variable(max_time, shape=(batch_size,), dtype=tf.int32)
input_states_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
input_states_ta = inputs_states_ta.unstack(input_states)


think: is the focus on the last words (given by the rnn) good?
why are we using the same weight matrix in similarity measure?
google would be better i think if it redirects you to the first page and has a small square in the bottom right with the other results.
how do you say something that is valid in all the meta versions? For instance, I can say: be consistant in life. then, that rule applies to itself. So you need to be consistant in being consistant. But, if I limit myself to some scope (eg be consistant in programming) then I'm not talking about the consistantcy about following the rule (which is interesting, because not being consistant for the rule produces a similar effect to not being consistant for programming)


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
