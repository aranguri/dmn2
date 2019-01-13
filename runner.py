import itertools
import tensorflow as tf
from preprocess import BabiTask
from model import DMNCell
import numpy as np
from utils import *

learning_rate = 1e-4
batch_size_train = 64
batch_size_dev = 300
embeddings_size = 256
h_size = 512
similarity_layer_size = 512
debug_steps = 50

babi_task = BabiTask(batch_size_train)
input_length, question_length, vocab_size = babi_task.get_lengths()

train = tf.placeholder(tf.bool, shape=())
input_ids = tf.placeholder(tf.int32, shape=(None, input_length))
question_ids = tf.placeholder(tf.int32, shape=(None, question_length))
supporting = tf.placeholder(tf.int32, shape=(None, None))

embeddings = tf.get_variable('embeddings', shape=(vocab_size, embeddings_size))
input = tf.nn.embedding_lookup(embeddings, input_ids)
question = tf.nn.embedding_lookup(embeddings, question_ids)
eos_vector = tf.nn.embedding_lookup(embeddings, babi_task.eos_vector)

dmn_cell = DMNCell(eos_vector, vocab_size, h_size, similarity_layer_size, learning_rate, optimize=train)
loss, accuracy, gates = dmn_cell.run(input, question, supporting)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tr_loss, tr_gates, dev_loss, dev_gates = {}, {}, {}, {}

    for j in itertools.count():
        input_, question_, answer_, sup_ = babi_task.next_batch()
        feed_dict = {input_ids: input_, question_ids: question_, supporting: sup_,
                     train: True}
        tr_loss[j], tr_gates[j], gates_ = sess.run([loss, accuracy, gates], feed_dict)

        if j % debug_steps == 0:
            input_, question_, answer_, sup_ = babi_task.dev_data()
            feed_dict = {input_ids: input_, question_ids: question_, supporting: sup_,
                         train: False}
            dev_loss[j/debug_steps], dev_gates[j/debug_steps], gates_ = sess.run([loss, accuracy, gates], feed_dict)

            tr_loss_, tr_gates_ = list(tr_loss.values()), list(tr_gates.values())
            dev_loss_, dev_gates_ = list(dev_loss.values()), list(dev_gates.values())
            print(f'{j}) TRAIN: Loss: {np.mean(tr_loss_[-10:])}. Gates: {np.mean(tr_gates_[-10:])}')
            print(f'{j}) DEV  : Loss: {np.mean(dev_loss_[-10:])}. Gates: {np.mean(dev_gates_[-10:])}')
            # smooth_plot(gates_acc)
            # smooth_plot(tr_loss)
