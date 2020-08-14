import pandas as pd
import numpy as np
import tensorflow as tf
import time

from preprocess import process_labeled_data

from models.baseline import train_validation_split
from models.sequential import vectorize_data, plot_accuracy_loss
from models.subword_encoder import BPE

NUM_EPOCHS = 10
EMBEDDING_DIM = 64




def read_preprocess():
    unprocessed = pd.read_csv('/Users/aksharma/PycharmProjects/Trump_Tweets/data/labeled_data.csv')

    preprocessed = process_labeled_data(unprocessed, stemming_lemmatize=False)

    x = preprocessed['text'].to_numpy()
    labels = preprocessed['sentiment'].to_numpy()

    tweets = np.array([' '.join(tweet) for tweet in x])

    return tweets, labels


def subword_tokenize(tweets, encoder):
    encoder.train_vocab()

    tokenized = []
    for tweet in tweets:
        sublist = []
        words = tweet.split()

        for word in words:
            sublist.extend(encoder.encode_word(word))

        encoded_tweet = ' '.join(sublist)
        tokenized.append(encoded_tweet)

    return tokenized


def create_recurrent_nn(vocab_size, embedding_dim, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    return model


def train_rnn(model, train_features, train_labels, test_features, test_labels):
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_features, train_labels, epochs=NUM_EPOCHS, batch_size=250,
                        validation_data=(test_features, test_labels), verbose=1)
    return model, history


def main():
    # Main for testing
    tweets, labels = read_preprocess()

    # encoding tweets
    encoder = BPE(tweets, num_merges=1000)
    tokenized = subword_tokenize(tweets, encoder)

    # spitting into train/test
    train_x, test_x, train_y, test_y = train_validation_split(tokenized, labels)
    padded_train, padded_test, max_length = vectorize_data(train_x, test_x, len(encoder.vocab))

    # creating model
    model = create_recurrent_nn(len(encoder.vocab), EMBEDDING_DIM, max_length)
    print(model.summary())

    # training model
    fitted_model, history = train_rnn(model, padded_train, train_y, padded_test, test_y)
    print(history)
    print(fitted_model.evaluate(padded_test, test_y))

    # accuracy loss
    plot_accuracy_loss(history, 'loss', NUM_EPOCHS)


if __name__ == '__main__':
    main()
