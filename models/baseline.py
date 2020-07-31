import numpy as np
import pandas as pd

from preprocess import process_labeled_data

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from sklearn.linear_model import LogisticRegression


def read_preprocessed_data():
    labeled_data = pd.read_csv(r'/Users/aksharma/PycharmProjects/Trump_Tweets/data/labeled_data.csv')

    preprocessed = process_labeled_data(labeled_data)

    x = preprocessed['text'].to_numpy()
    labels = preprocessed['sentiment'].to_numpy()

    tweets = np.array([' '.join(tweet) for tweet in x])

    return tweets, labels


def train_validation_split(tweets, labels):
    return train_test_split(tweets, labels, test_size=0.3, random_state=42)


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


def naive_bayes(tf_idf_train, train_labels):
    # train multinomial NB model
    classifier = MultinomialNB()
    classifier.fit(tf_idf_train, train_labels)

    return classifier


def logistic_regression(tf_idf_train, train_labels):
    # train logistic regression model
    classifier = LogisticRegression()
    classifier.fit(tf_idf_train, train_labels)

    return classifier


def compute_metrics(predicted_labels, actual_labels, model_type):
    # print('\n' + model_type + " statistics:")
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


def kfold_cross_validation(tweets, labels):
    kf = StratifiedKFold(n_splits=10, shuffle=True)

    i = 1
    for train_index, test_index in kf.split(tweets, labels):
        print('\n{} of kfold {}'.format(i, kf.n_splits))
        X_train, X_test = tweets[train_index], tweets[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # compute vectorized features
        tf_idf_train, tf_idf_test = vectorize_features(X_train, X_test, (1, 2))
        model = logistic_regression(tf_idf_train, y_train)

        print('Train Set Stats')
        compute_metrics(model.predict(tf_idf_train), y_train, 'K Fold')

        print('\n' + 'Test Set Stats')
        compute_metrics(model.predict(tf_idf_test), y_test, 'K Fold')
        i += 1


def main():
    # ML model with simple train_test split
    tweets, labels = read_preprocessed_data()
    train_x, test_x, train_y, test_y = train_validation_split(tweets, labels)

    train_features, test_features = vectorize_features(train_x, test_x, (1, 2))

    # naive_bayes
    classifier = naive_bayes(train_features, train_y)
    predicted_labels = classifier.predict(test_features)

    compute_metrics(predicted_labels, test_y, 'Naive Bayes')
    print(display_confusion_matrix(test_y, predicted_labels, 'Naive Bayes'))

    # logistic regression
    model = logistic_regression(train_features, train_y)
    logistic_labels = model.predict(test_features)

    compute_metrics(logistic_labels, test_y, 'Logistic Regression')
    print(display_confusion_matrix(test_y, logistic_labels, 'Logistic Regression'))

    # cross_validation
    kfold_cross_validation(tweets, labels)


if __name__ == '__main__':
    main()
