import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from models.preprocess import process_labeled_data
from models.sequential import train_tokenizer, vectorize_data, plot_accuracy_loss
from models.subword_encoder import BPE

# hyperparameters
NUM_EPOCHS = 10
EMBEDDING_DIM = 64


def read_preprocess():
    """
        Reads the labeled data csv file and preprocesses it with the routine from
        preprocess.py. Transforms each tweet into a normal string and maps both
        the tweet data and the labels into a numpy array.

            args: None
            ret:
                tweets: numpy array containing all the tweets
                labels: numpy array containing all the scalar sentiment labels
                (e.g 0 -> 'negative', 1 -> 'neutral', 2 -> 'positive')

    """
    unprocessed = pd.read_csv('/Users/aksharma/PycharmProjects/Trump_Tweets/data/labeled_data.csv')

    preprocessed = process_labeled_data(unprocessed, stemming_lemmatize=False)

    x = preprocessed['text'].to_numpy()
    labels = preprocessed['sentiment'].to_numpy()

    tweets = np.array([' '.join(tweet) for tweet in x])

    return tweets, labels


def subword_tokenize(tweets, encoder):
    """
        Encodes the tweets using byte pair encoding implemented in the BPE class.
        Assembles adjacent character level tokens over a period of merges, and uses
        the trained subword encoder to encode each word in the twitter data.

            args:
                tweets: all of the tweets
            ret:
                tokenized: all tweets that are encoded based on subword tokenization

    """
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


def train_validation_split(tokenized, labels):
    """
           Splits the data into train and validation sets with a 70/30 split. Also 'stratifies'
           in order to make sure that the train and validation sets contain equal proportions of
           positive/neutral/negative tweets.

               args:
                   tokenized: tweets that have been subword-encoded
                   labels: sentiment labels for each tweet in dataset
               ret:
                    train_x, test_x, train_y, test_y: train data and test data

    """
    return train_test_split(tokenized, labels, test_size=0.3, random_state=42, stratify=labels)


def create_recurrent_nn(vocab_size, embedding_dim, max_length):
    """
        Creates a LSTM with 6 total layers and using the input
        hyperparameters.

            args:
                vocab_size: total number of vocab features
                embedding_dim: length of word embeddings
                max_length: max length of any tweet
            ret:
                model: recurrent neural network (LSTM)

    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    return model


def train_rnn(model, train_features, train_labels, test_features, test_labels):
    """
        Trains the sequential neural network using rmsprop as the optimizer and
        sparse_categorical_crossentropy (probabilities) as the loss function.

            args:
                model: sequential model
                train_features: vectorized train data
                test_features: vectorized test data
            ret:
                model: the now fitted model on the train data
                history: documentation of accuracy/other metrics as the model
                fits on the number of epochs.

    """
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

    # vectorize the data
    fitted_tokenizer = train_tokenizer(train_x, vocab_size=len(encoder.vocab))
    padded_train, padded_test, max_length = vectorize_data(fitted_tokenizer=fitted_tokenizer, train_x=train_x,
                                                           test_x=test_x)

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
