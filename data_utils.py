import numpy as np
class CoNLLDataset(object):
    datasets = []
    def __init__(self, filename):
        with open(filename, encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                line = line.strip(' ').split(' ')
                self.datasets.append(line)

def add_glove(filename, dim_word):
    a = ['<start>']
    for i in range(dim_word):
        a.append('0.0')
    with open(filename, 'a+', encoding='utf-8') as f:
        f.write(' '.join(a))
        f.close()

def get_vocab(datasets):
    vocab = dict()
    vocab['<pad>'] = 0
    vocab['<start>'] = 1
    vocab['<eos>'] = 2
    vocab['<unk>'] = 3
    i = 4
    for line in datasets:
        for word in line:
            flag = vocab.get(word)
            if flag is None:
                vocab[word] = i
                i += 1
    return vocab

def load_vocab(filename):
    d = dict()
    with open(filename, encoding='utf-8') as f:
        i = 0
        for word in f:
            word = word.strip('\n')
            d[word] = i
            i += 1
    return d

def index2words(filename):
    d = dict()
    with open(filename, encoding='utf-8') as f:
        i = 0
        for word in f:
            word = word.strip('\n')
            d[i] = word
            i += 1
    return d

def write_vocab(filename, vocab):
    result = sorted(vocab.items(), key=lambda x: x[1], reverse=False)
    with open(filename, 'w', encoding='utf-8') as f:
        for word in result:
            f.write(word[0]+'\n')


def get_trimmed_datasets(filename, datasets, vocab, max_length):
    embeddings = np.zeros([len(datasets), max_length])
    k = 0
    for line in datasets:
        sen = np.zeros(max_length)
        sen[0] = vocab['<start>']
        for i in range(max_length-1):
            if i == max_length-2:
                sen[max_length-1] = vocab['<eos>']
                break
            if i == len(line):
                sen[i+1] = vocab['<eos>']
                break
            else:
                flag = vocab.get(line[i])
                if flag is None:
                    sen[i+1] = vocab['<unk>']
                else:
                    sen[i+1] = vocab[line[i]]
        embeddings[k] = sen
        k += 1
    np.savez_compressed(filename, test=embeddings)