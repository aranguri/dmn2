from functools import reduce
import re
import tarfile
import numpy as np

from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def parse_stories(lines, only_sup=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true,
    only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
            sup_acc = {}
            sup_acc[nid] = 1
        if '\t' in line:
            q, a, sup = line.split('\t')
            q = tokenize(q)
            substory = [x for x in story if x]

            sup = np.array([int(n) - sup_acc[int(n)] for n in sup.split()])
            data.append((substory, q, a, sup))
            story.append('')
            sup_acc[nid] = sup_acc[nid - 1] + 1
        else:
            if nid != 1:
                sup_acc[nid] = sup_acc[nid - 1]
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_sup=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_sup=only_sup)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer, sup) for story, q, answer, sup in data
            if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    sups = []

    for story, query, answer, sup in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
        sups.append(sup)

    return (pad_sequences(xs, maxlen=story_maxlen),
            pad_sequences(xqs, maxlen=query_maxlen),
            np.array(ys), np.array(sups))

def get_data():
    path = get_file('babi-tasks-v1-2.tar.gz',
                origin='https://s3.amazonaws.com/text-datasets/'
                       'babi_tasks_1-20_v1-2.tar.gz')

    # Default QA1 with 1000 samples
    # challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
    # QA1 with 10,000 samples
    # challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
    # QA2 with 1000 samples
    # challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
    # QA2 with 10,000 samples
    # challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'
    with tarfile.open(path) as tar:
        train = get_stories(tar.extractfile(challenge.format('train')))
        test = get_stories(tar.extractfile(challenge.format('test')))

    vocab = set()
    for story, q, answer, _ in train + test:
        vocab |= set(story + q + [answer])
    vocab = sorted(vocab)

    print(vocab)
    for t in train: print (t, '\n')
    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _, _ in train + test)))

    x, xq, y, sup = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
    tx, txq, ty, tsup = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

    # np.savez('babi/generated_data_one_fact_sup', x, xq, y, sup, tx, txq, ty, tsup, vocab_size, word_idx['.'])

get_data()
