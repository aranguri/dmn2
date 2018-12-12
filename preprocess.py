from pprint import pprint
import numpy as np
import pandas as pd
from utils import *

EMBEDDINGS_PATH = '../nns/datasets/glove/glove.6B.50d.txt'

class BabiTask:
    def __init__(self, task_num, batch_size):
        self.task_num = task_num
        self.batch_size = batch_size
        self.epoch = 0
        self.batch_ix = -1
        self.data = []

        self.load_embedding()
        self.load_data()
        self.preprocess_data()

    def load_embedding(self):
        #TODO: read from pickle instead of txt
        df = pd.read_csv(EMBEDDINGS_PATH, sep=' ', quoting=3, header=None, index_col=0)
        self.embed_weights = {key: val.values for key, val in df.T.items()}

    def load_data(self):
        input_ = []

        with open(f'bAbI-tasks/task_{self.task_num}.txt') as file:
            task = file.read()
            sentences = task.split('\n')
            sentences = np.array([np.array(s.split(' ')) for s in sentences])
            for s in sentences:
                # We want to see if any entry in s has the character
                #  '\t' which only answers have.
                if any(np.core.defchararray.find(s, '\t') != -1):
                    s = np.concatenate((s[:-1], s[-1].split('\t')))
                    s = [self.embed(w) for w in s]
                    # TODO: modify to support multiple label indices
                    # TODO: use np array
                    question, answer, label = s[0:-2], s[-2], s[-1]
                    input_ = np.array(input_)
                    self.data.append((input_, question, answer, label))
                    input_ = []
                else:
                    s = [self.embed(w) for w in s]
                    input_.append(s)

    def preprocess_data(self):
        data = np.array(self.data)
        num_batches = (len(data) // self.batch_size)
        data = data[:num_batches * self.batch_size]
        self.data = data.reshape(num_batches, self.batch_size, -1)
        self.num_batches = num_batches

    def next_batch(self):
        #TODO: better way to do this?
        self.batch_ix += 1
        if self.batch_ix > self.num_batches:
            self.batch_ix = -1
        return self.data[self.batch_ix]

    def embed(self, word):
        if word not in self.embed_weights.keys():
            dims = len(self.embed_weights)
            self.embed_weights[word] = np.random.randn(dims)
        return self.embed_weights[word]

    def eos_vector(self):
        return self.embeds('.')
