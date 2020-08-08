import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as plt
import re

from collections import defaultdict
from preprocess import process_labeled_data

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

NUM_EPOCHS = 10


def read_preprocess():
    unprocessed = pd.read_csv('/Users/aksharma/PycharmProjects/Trump_Tweets/data/labeled_data.csv')

    preprocessed = process_labeled_data(unprocessed, stemming_lemmatize=False)

    x = preprocessed['text'].to_numpy()
    labels = preprocessed['sentiment'].to_numpy()

    tweets = np.array([' '.join(tweet) for tweet in x])

    return tweets, labels


def create_vocab(tweets):
    vocab = defaultdict(int)

    for tweet in tweets:
        for word in tweet.split():
            vocab[' '.join(list(word)) + ' <w>'] += 1

    return vocab


def get_pair_freqs(vocab):
    pairs = defaultdict(int)

    for word, freq in vocab.items():
        chars = word.split()
        for i in range(len(chars) - 1):
            pairs[chars[i], chars[i + 1]] += freq

    return pairs


def merge_vocab(pair, prev_vocab):
    # merges vocab into new vocab by finding max amount of one pair
    new_vocab = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

    # generate new vocab dictionary based on merged pair
    for word in prev_vocab:
        w_out = p.sub(''.join(pair), word)
        new_vocab[w_out] = prev_vocab[word]

    return new_vocab


def get_tokens(vocab):
    # returns individual characters or n-grams (total data tokens)
    # after each merge
    tokens = defaultdict(int)

    for word, freq in vocab.items():
        for token in word.split():
            tokens[token] += freq

    return tokens


def byte_pair_encoding(vocab, num_merges):
    for i in range(num_merges):
        pairs = get_pair_freqs(vocab)
        if not pairs:
            break

        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        tokens = get_tokens(vocab)
        print('Iteration: {}'.format(i))
        print('Best Pair: {}'.format(best))
        print('Number of Tokens: {}'.format(len(tokens)))


def encode_word(word, tokens):
    pass


def vectorize_data(train_data, test_data):
    pass


def create_recurrent_nn(vocab_size, embedding_dim, max_length):
    pass


def train_rnn(model, train_features, train_labels, test_features, test_labels):
    pass


def plot_accuracy_loss(model, history):
    pass


def main():
    tweets, _ = read_preprocess()

    vocab = create_vocab(tweets)

    byte_pair_encoding(vocab, 1000)


if __name__ == '__main__':
    main()
