import numpy as np
import pandas as pd
import os.path
from utils import *

class BabiGen:
    def __init__(self, task_num, batch_size, embeddings_size):
        self.task_num = task_num
        self.batch_size = batch_size
        self.embeddings_size = embeddings_size
        self.data = []
        self.embed_weights = {}

        self.load_data()
        self.preprocess_data()
        # self.store_data()

    def load_data(self):
        input = []

        with open(f'babi/task_{self.task_num}_back.txt') as file:
            task = file.read()
            task = task.replace('.', ' .')
            task = task.replace('?', ' ?')
            sentences = task.split('\n')
            sentences = np.array([np.array(s.split(' ')) for s in sentences])

            # Prevent reading last line, because atom added one new line.
            for s in sentences[:-1]:
                if s[0] == '1':
                    input = []

                # We want to see if any entry in s has the character
                #  '\t' which only answers have.
                if any(np.core.defchararray.find(s, '\t') != -1):
                    s = np.concatenate((s[:-1], s[-1].split('\t')))
                    s = [self.embed(w) for w in s]
                    # TODo: modify to support multiple label indices
                    question, answer, label = s[0:-2], s[-2], s[-1]
                    input_flatten = np.array(input).reshape(-1, self.embeddings_size)
                    self.data.append((input_flatten, question, answer, label))
                else:
                    # TODo: modify to store sentence number
                    s = [self.embed(w) for w in s[1:]]
                    input.append(s)

    def preprocess_data(self):
        data = np.array(self.data)
        num_batches = len(data) // self.batch_size
        data = data[:num_batches * self.batch_size]
        data = data.reshape(num_batches, self.batch_size, -1)
        data = np.swapaxes(data, 1, 2)

        # Sorry world for the fors.
        for i in range(len(data)):
            max_length = max([data[i, 0, j].shape[0] for j in range(len(data[i, 0]))])
            for j in range(len(data[i, 0])):
                d = data[i, 0, j]
                if max_length != d.shape[0]:
                    ps(np.pad(d, ((0, max_length - d.shape[0]), (0, 0)), 'constant'))
                    data[i, 0, j] = np.pad(d, ((0, max_length - d.shape[0]), (0, 0)), 'constant')
        return data
        self.num_batches = num_batches

    def store_data(self):
        file_name = f'babi/parsed/{self.task_num}_{self.batch_size}'
        np.savez(file_name, self.data, self.embed('.'))

    def embed(self, word):
        if word not in self.embed_weights.keys():
            self.embed_weights[word] = np.random.randn(self.embeddings_size)
        return self.embed_weights[word]

    def get_data(self):
        return self.data

class BabiTask:
    def __init__(self, batch_size):
        self.epoch = 0
        self.i = -1 # Batch index
        self.batch_size = batch_size

        file_name = f'babi/generated_data_one_fact_sup_100.npz'
        file = np.load(file_name)

        self.x, self.xq, self.y, self.sup = file['arr_0'], file['arr_1'], file['arr_2'], file['arr_3']
        self.tx, self.txq, self.ty, tsup = file['arr_4'], file['arr_5'], file['arr_6'], file['arr_7']
        self.vocab_size = file['arr_8']
        self.eos_vector = file['arr_9']

    def get_lengths(self):
        return self.x.shape[1], self.xq.shape[1], self.vocab_size

    def next_batch(self):
        if (self.i + 2) * self.batch_size > len(self.x):
            self.i = 0
            self.epoch += 1
        else:
            self.i += 1

        return (self.x[self.i * self.batch_size:(self.i + 1) * self.batch_size],
                self.xq[self.i * self.batch_size:(self.i + 1) * self.batch_size],
                self.y[self.i * self.batch_size:(self.i + 1) * self.batch_size],
                self.sup[self.i * self.batch_size:(self.i + 1) * self.batch_size])

    def dev_data(self):
        return self.tx[:self.batch_size], self.txq[:self.batch_size], self.ty[:self.batch_size]
