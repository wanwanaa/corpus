from config import Config
from data_utils import get_datasets, get_train_vocab, get_glove_vocab, word2index, index2word, write_vocab,glove_embedding, get_trimmed_datasets


def main():
    # get config
    config = Config(load=False)

    # Generators
    train = get_datasets(config.filename_train)
    valid = get_datasets(config.filename_valid)
    test = get_datasets(config.filename_test)

    # add <start> to glove
    # add_glove(config.filename_glove, config.dim_word)

    # Build word vocab
    train_words = get_train_vocab(train)
    glove_vocab = get_glove_vocab(config.filename_glove)

    # train & glove(word to index)
    vocab = word2index(train_words, glove_vocab)
    # save vocab
    write_vocab(config.filename_words, vocab)

    # index to word
    index = index2word(vocab)
    write_vocab(config.filename_index, index)

    # embedding
    glove_embedding(config.filename_glove, config.filename_trimmed_glove, config.dim_word, vocab, config.start, config.pad)

    # trim datasets
    get_trimmed_datasets(config.filename_trimmed_train, train, vocab, config.max_length)
    get_trimmed_datasets(config.filename_trimmed_valid, valid, vocab, config.max_length)
    get_trimmed_datasets(config.filename_trimmed_test, test, vocab, config.max_length)


if __name__ == '__main__':
    main()