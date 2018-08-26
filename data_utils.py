import numpy as np
import pickle
'''class CoNLLDataset(object):
    datasets = []
    def __init__(self, filename):
        with open(filename, encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                line = line.strip(' ').split(' ')
                self.datasets.append(line)'''


def get_datasets(filename):
    dataset = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            line = line.strip(' ').split(' ')
            dataset.append(line)
    return dataset

def add_glove(filename, dim_word):
    a = ['<start>']
    for i in range(dim_word):
        a.append('0.0')
    with open(filename, 'a+', encoding='utf-8') as f:
        f.write(' '.join(a))
        f.close()


'''def get_vocab(datasets):
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
    return vocab'''


def get_train_vocab(dataset):
    vocab = set()
    for line in dataset:
        for word in line:
            vocab.add(word)
    return vocab


def get_glove_vocab(filename):
    vocab = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split(' ')
            vocab.add(line[0])
    return vocab


def word2index(train_words, glove_vocab):
    words = train_words & glove_vocab
    words = list(words)
    vocab = dict()
    vocab['<pad>'] = 0
    vocab['<start>'] = 1
    vocab['eos'] = 2
    vocab['unk'] = 3
    i = 4
    for word in words:
        flag = vocab.get(word)
        if flag is None:
            vocab[word] = i
            i += 1
    return vocab


def index2word(vocab):
    index = []
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=False)
    for i in vocab:
        index.append(i[0])
    return index


def glove_embedding(filename_glove, filename_trimmed_glove, dim_word, vocab, start, pad):
    embeddings = dict()
    with open(filename_glove, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            if word in vocab.keys():
                embedding = [float(x) for x in line[1:]]
                embeddings[vocab[word]] = embedding
    if pad == 1:
        embedding = np.ones(dim_word)
        embeddings[0] = embedding
    if start == 1:
        embedding = np.zeros(dim_word)
        embeddings[1] = embedding
    embeddings = sorted(embeddings.items(), key=lambda x: x[0], reverse=False)
    embeddings_array = np.zeros((embeddings[-1][0]+1, dim_word))
    for i in embeddings:
        embeddings_array[i[0]] = i[1]
    np.savez_compressed(filename_trimmed_glove, embeddings=embeddings_array)


def write_vocab(filename, vocab):
    with open(filename, 'wb') as f:
        pickle.dump(vocab, f)


'''def load_vocab(filename):
    d = dict()
    with open(filename, encoding='utf-8') as f:
        i = 0
        for word in f:
            word = word.strip('\n')
            d[word] = i
            i += 1
    return d'''

'''def index2words(filename):
    d = dict()
    with open(filename, encoding='utf-8') as f:
        i = 0
        for word in f:
            word = word.strip('\n')
            d[i] = word
            i += 1
    return d'''

'''def write_vocab(filename, vocab):
    result = sorted(vocab.items(), key=lambda x: x[1], reverse=False)
    with open(filename, 'w', encoding='utf-8') as f:
        for word in result:
            f.write(word[0]+'\n')'''


def get_trimmed_datasets(filename, datasets, vocab, max_length):
    embeddings = np.zeros([len(datasets), max_length])
    k = 0
    for line in datasets:
        sen = np.zeros(max_length)
        sen[0] = vocab['<start>']
        for i in range(max_length-1):
            if i == max_length-2:
                sen[max_length-1] = vocab['eos']
                break
            if i == len(line):
                sen[i+1] = vocab['eos']
                break
            else:
                flag = vocab.get(line[i])
                if flag is None:
                    sen[i+1] = vocab['unk']
                else:
                    sen[i+1] = vocab[line[i]]
        embeddings[k] = sen
        k += 1
    np.savez_compressed(filename, index=embeddings)
