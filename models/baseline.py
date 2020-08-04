import numpy as np
import pandas as pd

from preprocess import process_labeled_data

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def read_labeled_data():
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

    labeled_data = pd.read_csv(r'/Users/aksharma/PycharmProjects/Trump_Tweets/data/labeled_data.csv')

    preprocessed = process_labeled_data(labeled_data)

    x = preprocessed['text'].to_numpy()
    labels = preprocessed['sentiment'].to_numpy()

    tweets = np.array([' '.join(tweet) for tweet in x])

    return tweets, labels


def get_class_proportions(labels):
    """
        Gets the proportion (ratio) of each label class in the preprocessed dataset.
        Used to make sure that splitting of train set and validation set is not uneven
        and contains a equal percentage of each sentiment class.

            args:
                labels: sentiment labels for each tweet in dataset
            ret:
                class_proportions: dict containing a mapping from each label to its proportion

    """

    classes = {0: 'negative', 1: 'neutral', 2: 'positive'}

    return {val[1]: (labels == val[0]).mean() for val in classes.items()}


def train_validation_split(tweets, labels):
    """
        Splits the data into train and validation sets with a 70/30 split. Also 'stratifies'
        in order to make sure that the train and validation sets contain equal proportions of
        positive/neutral/negative tweets.

            args:
                labels: sentiment labels for each tweet in dataset
            ret:
                class_proportions: dict containing a mapping from each label to its proportion

    """

    return train_test_split(tweets, labels, test_size=0.3, random_state=42, stratify=labels)


def vectorize_features(train_x, test_x, grams):
    """
        Transforms the raw train and test data into vectorized tf_idf representations. Tabulates
        features (2000 most common 'grams') by fitting on the train_data to the learn a vocabulary.
        Computes both the term frequency (bag of words) and balances it by inverse document frequency
        for each tweet in the corpus. Transforms both the train_data and test_data to generate the
        tf_idf representation.

             args:
                train_x: train data
                test_x: test data
                grams: tuple that specifies what range of words to take as features
                (e.g (1, 2) -> both unigrams and bigrams)
             ret:
                tf_idf_train: tf_idf representation of train data, fitted to learn vocab and transformed
                tf_idf_test: tf_idf representation of test_data, only transformed

    """

    # compute bag of words vectorization
    vectorizer = CountVectorizer(max_features=2000, min_df=7, max_df=0.8, ngram_range=grams)
    bag_of_words_train = vectorizer.fit_transform(train_x)

    bag_of_words_test = vectorizer.transform(test_x)

    # compute tf_idf vectorization
    transformer = TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=True)
    tf_idf_train = transformer.fit_transform(bag_of_words_train)

    tf_idf_test = transformer.transform(bag_of_words_test)

    return tf_idf_train, tf_idf_test


def create_naive_bayes(tf_idf_train, train_labels):
    """
        creates and fits the naive bayes regression (multinomial NB) on the
        train data.

            args:
                tf_idf_train: vectorized train data
                train_labels: sentiment class for each tweet
            ret:
                classifier: naive bayes classifier (Multinomial NB)

    """

    # train multinomial NB model
    classifier = MultinomialNB()
    classifier.fit(tf_idf_train, train_labels)

    return classifier


def create_logistic_regression(tf_idf_train, train_labels):
    """
        creates and fits the logistic regression on the train data.

            args:
                tf_idf_train: vectorized train data
                train_labels: sentiment class for each tweet
            ret:
                classifier: logistic regression model

    """

    # train logistic regression model
    classifier = LogisticRegression()
    classifier.fit(tf_idf_train, train_labels)

    return classifier


def compute_metrics(predicted_labels, actual_labels, model_type):
    """
        Displays relevant metrics of the ML model on the test data, which include:
        Accuracy, F1 Score, Precision, Recall, and the Confusion Matrix.

            args:
                predicted_labels: labels predicted by the ML model
                actual_labels: original labels
                model_type: model name (either logistic regression or naive bayes)
            ret:
                None: prints the metrics

    """

    print('\n' + model_type + '\n' + 'Accuracy:', np.round(
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

    print('\n' + model_type + ' confusion matrix:')
    print(metrics.confusion_matrix(actual_labels, predicted_labels, labels=[0, 1, 2]))


def kfold_cross_validation(tweets, labels):
    """
        Trains the model with stratified K_Fold cross validation. Train data is partitioned into
        k = 10 different subsets and model is fit on one of the subsets on each iteration. Used for
        computing whether the model is overfitting/underfitting on the train_data by comparing training
        and validation, and the overall skill of the model. Computes all the metrics on each iteration.

            args:
                predicted_labels: labels predicted by the ML model
                actual_labels: original labels
                model_type: model name (either logistic regression or naive bayes)
            ret:
                    None: prints the metrics

    """

    kf = StratifiedKFold(n_splits=10, shuffle=True)

    i = 1
    for train_index, test_index in kf.split(tweets, labels):
        print('\n{} of kfold {}'.format(i, kf.n_splits))
        X_train, X_test = tweets[train_index], tweets[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # compute vectorized features
        tf_idf_train, tf_idf_test = vectorize_features(X_train, X_test, (1, 2))
        model = create_logistic_regression(tf_idf_train, y_train)

        print('\n' + 'Train Set Stats')
        compute_metrics(model.predict(tf_idf_train), y_train, 'K Fold')

        print('\n' + 'Test Set Stats')
        compute_metrics(model.predict(tf_idf_test), y_test, 'K Fold')
        i += 1


def main():
    # ML model with simple train_test split
    tweets, labels = read_labeled_data()
    train_x, test_x, train_y, test_y = train_validation_split(tweets, labels)

    train_features, test_features = vectorize_features(train_x, test_x, (1, 2))

    # naive_bayes
    classifier = create_naive_bayes(train_features, train_y)
    predicted_labels = classifier.predict(test_features)

    compute_metrics(predicted_labels, test_y, 'Naive Bayes')

    # logistic regression
    model = create_logistic_regression(train_features, train_y)
    logistic_labels = model.predict(test_features)

    compute_metrics(logistic_labels, test_y, 'Logistic Regression')

    # cross_validation
    kfold_cross_validation(tweets, labels)


if __name__ == '__main__':
    main()
