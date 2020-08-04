import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models.baseline import read_labeled_data, train_validation_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

NUM_EPOCHS = 30
VOCAB_SIZE = 10000
EMBEDDING_DIMENSIONS = 8


def read_split_data():
    tweets, labels = read_labeled_data()
    return train_validation_split(tweets, labels)


def vectorize_data(train_x, test_x, vocab_size):
    max_length = max([len(tweet.split()) for tweet in np.concatenate((train_x, test_x), axis=None)])

    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_x)

    training_sequences = tokenizer.texts_to_sequences(train_x)
    padding_train = pad_sequences(training_sequences, maxlen=max_length, padding='post')

    testing_sequences = tokenizer.texts_to_sequences(test_x)
    padding_test = pad_sequences(testing_sequences, maxlen=max_length, padding='post')

    training_padded = np.array(padding_train)
    testing_padded = np.array(padding_test)

    return training_padded, testing_padded, max_length


def encode_labels(train_y, test_y):
    return to_categorical(train_y), to_categorical(test_y)


def create_sequential_nn(vocab_size, embedding_dim, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    return model


def train_fnn(model, train_features, train_labels, test_features, test_labels):
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_features, train_labels, epochs=NUM_EPOCHS, batch_size=250,
                        validation_data=(test_features, test_labels), verbose=1)
    return model, history


def get_model_weights(model):
    return model.layers[0].get_weights()[0]


def plot_accuracy_loss(history, metric_type, epochs):
    metric = history.history[metric_type]
    validation_metric = history.history['val_' + metric_type]

    line = range(1, epochs + 1)

    plt.plot(line, metric, 'bo', label='Train ' + metric_type)
    plt.plot(line, validation_metric, 'b', label='Validation ' + metric_type)

    plt.legend()
    plt.show()


def main():
    train_x, test_x, train_y, test_y = read_split_data()

    padded_train, padded_test, max_length = vectorize_data(train_x, test_x, VOCAB_SIZE)

    # encoded_train, encoded_test = encode_labels(train_y, test_y)

    model = create_sequential_nn(VOCAB_SIZE, EMBEDDING_DIMENSIONS, max_length)
    print(model.summary())
    fitted_model, history = train_fnn(model, padded_train, train_y, padded_test, test_y)

    print(history)
    print(fitted_model.evaluate(padded_test, test_y))

    plot_accuracy_loss(history, 'accuracy', NUM_EPOCHS)


if __name__ == '__main__':
    main()
