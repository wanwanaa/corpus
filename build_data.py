from config import Config
from data_utils import CoNLLDataset, add_glove, get_vocab, write_vocab, load_vocab, get_trimmed_datasets


def main():
    # get config
    config = Config(load=False)

    # Generators
    test = CoNLLDataset(config.filename_test)

    # add <start> to glove
    # nadd_glove(config.filename_glove, config.dim_word)

    # Build word vocab
    # vocab_words = get_vocab(test.datasets)

    # Save vocab
    # write_vocab(config.filename_words, vocab_words)

    # load vocab
    vocab_words = load_vocab(config.filename_words)

    # trim datasets
    get_trimmed_datasets(config.filename_trimmed, test.datasets, vocab_words, config.max_length)


if __name__ == '__main__':
    main()