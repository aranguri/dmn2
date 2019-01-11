import itertools
import tensorflow as tf
from preprocess import BabiTask
from model import DMNCell
import numpy as np
from utils import *

batch_size = 64
embeddings_size = 256

babi_task = BabiTask(batch_size)
input_length, question_length, vocab_size = babi_task.get_lengths()

input_ids = tf.placeholder(tf.int32, shape=(batch_size, input_length))
question_ids = tf.placeholder(tf.int32, shape=(batch_size, question_length))
supporting = tf.placeholder(tf.int32, shape=(batch_size, None))

embeddings = tf.get_variable('embedding', shape=(vocab_size, embeddings_size))
input = tf.nn.embedding_lookup(embeddings, input_ids)
question = tf.nn.embedding_lookup(embeddings, question_ids)
eos_vector = tf.nn.embedding_lookup(embeddings, babi_task.eos_vector)

dmn_cell = DMNCell(eos_vector, vocab_size, h_size=512, similarity_layer_size=512,
                   learning_rate=1e-4)

loss, accuracy, gates = dmn_cell.run(input, question, supporting)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tr_loss, tr_acc, dev_loss, dev_acc, gates_acc = {}, {}, {}, {}, {}

    for j in itertools.count():
        input_, question_, answer_, sup_ = babi_task.next_batch()
        feed_dict = {input_ids: input_, question_ids: question_,
                     supporting: sup_}
        tr_loss[j], tr_acc[j], gates_ = sess.run([loss, accuracy, gates], feed_dict)
        gates_acc[j] = np.mean(np.argmax(gates_, axis=2) == sup_)

        if j % 10 == 0:
            # print_all_vars(sess)
            for i, k in zip(np.argmax(gates_, axis=2), sup_):
                print(i, k)
            gates_acc_ = list(gates_acc.values())
            print(f'{j}) Mean: {np.mean(gates_acc_)}. Last 10: {np.mean(gates_acc_[-10:])}')
            # smooth_plot(gates_acc)
            # smooth_plot(tr_loss)

        '''
        # print(f'Loss: {tr_loss[j]}. Accuracy: {tr_acc[j]}. Output: {np.argmax(output_, axis=1)[:15]}')
        if j % 10 == 0:
            input_, question_, answer_ = babi_task.dev_data()
            feed_dict = {input_ids: input_, question_ids: question_, answer: answer_, optimize: False}
            dev_loss[j], dev_acc[j], output_ = sess.run([loss, accuracy, output], feed_dict)
            print(f'** Dev: Loss: {dev_loss[j]}. Accuracy: {dev_acc[j]}. **')
        '''
