import re
import json

import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


def read_json_data():
    """
         Reads and cleans data from the input dataset, which is in JSON form. Converts this JSON data into a pandas
         dataframe, removes rows that are retweets or that contain URLS, and drops the columns that aren't relevant
         (retweet_count, favorites, id, is_retweet).
         args: None
         ret:
             tweet_data: preprocessed twitter data according to the above conventions
    """

    with open('data/trump_tweets.json') as f:
        json_data = json.load(f)

    normalized_json = pd.json_normalize(json_data)

    # turn tweet data into pandas dataframe
    tweet_data = pd.DataFrame(normalized_json)

    # drop irrelevant parts of dataframe
    tweet_data = tweet_data[tweet_data.is_retweet == False]
    tweet_data = tweet_data.drop(['retweet_count', 'id_str', 'favorite_count', 'is_retweet'], axis=1).drop_duplicates(
        ['text'])

    # drop rows of dataframe that contain URLS or RTS
    tweet_data = tweet_data[~tweet_data['text'].str.contains('https://t.co') & ~tweet_data['text'].str.contains('RT')]

    # change &amp to just &
    tweet_data['text'] = tweet_data['text'].str.replace('&amp;', '&')

    # standardize datetime
    tweet_data['created_at'] = pd.to_datetime(tweet_data['created_at'])

    # remove multi-line tweets
    tweet_data = tweet_data[~tweet_data['text'].str.startswith('..') & ~tweet_data['text'].str.endswith('...')]

    # remove tweets that are simply quoted text from other people
    tweet_data = tweet_data[~tweet_data['text'].str.startswith('"')]

    return tweet_data


def pre_label_data(data):
    """
           Naively label sentiment data based on the presence of a list of words (e.g hillary -> negative,
           biden -> negative, rally -> positive, Make America Great Again -> positive).
           args:
               data: preprocessed pandas dataframe containing 3 columns: source, text, time
           ret:
              None: dataframe is prelabeled with sentiment values based on presence of certain words
    """
    neg_words = ['hillary', 'biden', 'joe', 'pelosi', 'schumer', 'democrat', 'fake news', 'obama', 'jeb', 'cnn',
                 'msnbc']
    pos_words = ['rally', 'make america great again', 'crowd']

    lower_case_data = data['text'].str.lower()

    data['sentiment'] = np.where(lower_case_data.str.contains('|'.join(neg_words)), 'negative', np.nan)
    data['sentiment'] = np.where(
        lower_case_data.str.contains('|'.join(pos_words)) & ~data['sentiment'].str.contains('negative'),
        'positive', data['sentiment'])


def process_tweet(tweet, stemming_lemmatize: bool):
    """
        "Clean" a tweet by retrieving only the most important parts required for NLP analysis. This includes the
        following steps:

        1. remove all text in quotations (not tweeted by trump), all irrelevant punctuation, and mentions
        2. tokenizing the tweet (splitting into individual words)
        3. Either stemming/lemmatizing words or not (for use in subword tokenization)

        args:
            tweet: unprocessed DJT tweet
        ret:
            processed tweet: tweet that has been cleaned
    """

    # remove all quoted text
    tweet = re.sub('"[^"]*"', '', tweet)

    # remove all mentions from text
    tweet = re.sub("@[A-Za-z0-9]+", "", tweet)

    # remove unnecessary punctuation and numbers
    tweet = re.sub('[^a-zA-Z]', ' ', tweet)

    # tokenize the tweet
    tokenized_tweet = word_tokenize(tweet.lower())

    # remove stopwords from tokenized tweet
    stopwords_set = set(stopwords.words('english'))
    tokenized_tweet = [word for word in tokenized_tweet if word not in stopwords_set]

    # Stemming and Lemmatization
    if stemming_lemmatize:
        stemmer = PorterStemmer()
        lem = WordNetLemmatizer()

        tokenized_tweet = [lem.lemmatize(word) for word in tokenized_tweet]
        tokenized_tweet = [stemmer.stem(word) for word in tokenized_tweet]

    return tokenized_tweet


def process_labeled_data(data, stemming_lemmatize: bool):
    """
           Apply the routine in process_tweet to the corpus of labeled tweets. After processing
           the dataset, remove the extra column (unnamed), drop all rows with NaN sentiment,
           and vectorize sentiment (1 -> negative, 2 -> neutral, 3 -> positive)

           args:
               data: unprocessed labeled data
           ret:
               data: labeled data that has been properly cleaned with respect to all the NLP conventions
    """
    data['text'] = data['text'].apply(lambda x: process_tweet(x, stemming_lemmatize=stemming_lemmatize))

    # remove all rows that have nan sentiment and drop extra column
    data = data.drop(['Unnamed: 0'], axis=1).dropna(subset=['sentiment'])

    # remove all rows that have < 2 words
    data = data[data['text'].str.len() > 2]

    # vectorize sentiment values
    data['sentiment'] = data['sentiment'].apply(lambda x: 0 if x == 'negative' else 1 if x == 'neutral' else 2)

    data.to_csv(r'/Users/aksharma/PycharmProjects/Trump_Tweets/Data/preprocessed_data.csv', index=False)

    return data


def main():
    tweet_data = read_json_data()

    pre_label_data(tweet_data)

    updated_data = pd.read_csv('data/labeled_data.csv')

    cleaned_data = process_labeled_data(updated_data, stemming_lemmatize=True)
    print(cleaned_data['text'])


if __name__ == '__main__':
    main()
