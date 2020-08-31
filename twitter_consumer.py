import numpy as np

import requests
import json
import re
import os

from models.preprocess import process_tweet
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.sequential import read_split_data, train_tokenizer

from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, CategoriesOptions
from ibm_watson import ApiException

VOCAB_SIZE = 10000

# hosted model on Microsoft Azure
PERSONAL_MODEL_URL = os.environ.get('SENTIMENT_API')

# authenticate natural language api
IBM_KEY = os.environ.get('IBM_KEY')
authenticator = IAMAuthenticator(IBM_KEY)
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2019-07-12',
    authenticator=authenticator
)

IBM_URL = os.environ.get('IBM_URL')

natural_language_understanding.set_service_url(IBM_URL)


def entity_recognition(tweet):
    if len(tweet.split()) < 3:
        return None

    # attempt to query IBM's natural language processing API for company entities
    try:
        response = natural_language_understanding.analyze(text=tweet, features=Features(entities=EntitiesOptions())) \
            .get_result()
    except ApiException:
        return None

    return response['entities'] if 'entities' in response else None


def obtain_companies(entities):
    return [entity['text'] for entity in entities if entity['type'] == 'Company'] if entities else None


def category_recognition(tweet):
    if len(tweet.split()) < 3:
        return None

    # try querying for general economic entities
    try:
        response = natural_language_understanding.analyze(
            text=tweet,
            features=Features(categories=CategoriesOptions(limit=3))).get_result()
    except ApiException:
        return None

    # find if the tweet is about unemployment
    for category in response['categories']:
        if category['label'] == '/society/work/unemployment':
            return ['$SPY']

    return None


def preprocess_tweet(tweet):
    # change &amp to and so we can use stopwords
    tweet = tweet.replace('&amp', 'and')

    # replace RTS
    tweet = tweet.replace('RT', '')

    # remove Links
    tweet = re.sub(r"http\S+", "", tweet)

    return [' '.join(process_tweet(tweet, stemming_lemmatize=True))]


def transform_processed(preprocessed):
    train_x, test_x, _, _ = read_split_data()

    max_length = max([len(tweet.split()) for tweet in np.concatenate((train_x, test_x), axis=None)])
    tokenizer = train_tokenizer(train_x, VOCAB_SIZE)

    sequences = tokenizer.texts_to_sequences(preprocessed)
    vectorized = pad_sequences(sequences, maxlen=max_length, padding='post')

    return vectorized.tolist()


def predict_sentiment(tweet):
    # preprocess tweet
    preprocessed = preprocess_tweet(tweet)
    vectorized = transform_processed(preprocessed)

    data = json.dumps({'signature_name': 'serving_default', 'instances': vectorized})

    headers = {"content-type": "application/json"}

    sentiment_prob = requests.post(PERSONAL_MODEL_URL, data=data, headers=headers).json()['predictions'][0]

    # find correct class (label)
    label = -1
    for i in range(len(sentiment_prob)):
        if sentiment_prob[i] > 0.5:
            label = i

    if label == -1:
        return None
    else:
        if label == 0:
            return 'negative'
        elif label == 1:
            return 'neutral'
        else:
            return 'positive'


if __name__ == "__main__":
    print(predict_sentiment(
        'It WOULD BE HORRIBLE FOR THE DO NOTHING DEMOCRATS TO GET ELECTED. SLEEPY JOE CAN BARELY FORM A SENTENCE. '
        'ELECT ME AND WE WILL WIN BIGLY!'))
