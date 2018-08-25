class Config():
    def __init__(self, load=True):
        if load:
            self.load()
    def load(self):
        pass

    # embeding
    dim_word = 300

    # words files
    filename_words = 'data/words.txt'
    # glove files
    filename_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    # trimmed embedding
    filename_trimmed = 'data/test.npz'

    # dataset
    # filename_train = 'data/test'
    # filename_dev = 'data/test'
    # filename_test = 'data/test'
    filename_train = filename_dev = filename_test = 'data/test'

    max_length = 32

    # vocab
    filename_words = 'data/words.txt'

    # training

    #model hyperparameters

