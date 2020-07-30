import numpy as np
import pandas as pd

from preprocess import process_labeled_data

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from sklearn.linear_model import LogisticRegression


def read_preprocessed_data():
    labeled_data = pd.read_csv('../data/labeled_data.csv')

    preprocessed = process_labeled_data(labeled_data)

    x = preprocessed['text'].to_numpy()
    labels = preprocessed['sentiment'].to_numpy()

    x = [' '.join(tweet) for tweet in x]

    train_x, test_x, train_y, test_y = train_test_split(x, labels, test_size=0.3, random_state=42)

    return train_x, test_x, train_y, test_y


def vectorize_features(train_x, test_x, grams):
    # compute bag of words vectorization
    vectorizer = CountVectorizer(max_features=2000, min_df=7, max_df=0.8, ngram_range=grams)
    bag_of_words_train = vectorizer.fit_transform(train_x)

    bag_of_words_test = vectorizer.transform(test_x)

    # compute tf_idf vectorization
    transformer = TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=True)
    tf_idf_train = transformer.fit_transform(bag_of_words_train)

    tf_idf_test = transformer.transform(bag_of_words_test)

    return tf_idf_train, tf_idf_test


def train_predict_naive_bayes(train_x, train_y, test_x):
    # train multinomial NB model
    classifier = MultinomialNB()
    classifier.fit(train_x, train_y)

    # compute predictions
    predictions = classifier.predict(test_x)
    return predictions


def train_predict_logistic_regression(train_x, train_y, test_x):
    # train logistic regression model
    classifer = LogisticRegression()
    classifer.fit(train_x, train_y)

    # compute predictions
    return classifer.predict(test_x)


def compute_metrics(predicted_labels, actual_labels, model_type):
    print('\n' + model_type + " statistics:")
    print('Accuracy:', np.round(
        metrics.accuracy_score(actual_labels, predicted_labels), 4
    ))

    print('Precision: ', np.round(
        metrics.precision_score(actual_labels, predicted_labels, average='weighted'), 4
    ))

    print('Recall:', np.round(
        metrics.recall_score(actual_labels, predicted_labels, average='weighted'), 4
    ))

    print('F1 Score:', np.round(
        metrics.f1_score(actual_labels, predicted_labels, average='weighted'), 4
    ))


def display_confusion_matrix(actual_labels, predicted_labels, model_type):
    print('\n' + model_type + ' confusion matrix: ')
    return metrics.confusion_matrix(actual_labels, predicted_labels, labels=[1, 2, 3])


def main():
    train_x, test_x, train_y, test_y = read_preprocessed_data()

    train_features, test_features = vectorize_features(train_x, test_x, (1, 2))

    # naive_bayes
    predicted_labels = train_predict_naive_bayes(train_features, train_y, test_features)
    compute_metrics(predicted_labels, test_y, 'Naive Bayes')
    print(display_confusion_matrix(test_y, predicted_labels, 'Naive Bayes'))

    # logistic regression
    logistic_labels = train_predict_logistic_regression(train_features, train_y, test_features)
    compute_metrics(logistic_labels, test_y, 'Logistic Regres   sion')
    print(display_confusion_matrix(test_y, logistic_labels, 'Logistic Regression'))


if __name__ == '__main__':
    main()
