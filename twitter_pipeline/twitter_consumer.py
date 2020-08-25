import numpy as np

import requests
import json
import re
import os

from preprocess import process_tweet
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.sequential import read_split_data, train_tokenizer

from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, ConceptsOptions

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

    try:
        response = natural_language_understanding.analyze(text=tweet, features=Features(entities=EntitiesOptions())) \
            .get_result()
    except Exception:
        return None

    return response['entities'] if 'entities' in response else None


def obtain_companies(entities):
    return [entity['text'] for entity in entities if entity['type'] == 'Company']


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


def predict_sentiment(tweet_info):
    tweet = tweet_info['tweet']

    entities = entity_recognition(tweet)

    if entities:
        companies = obtain_companies(entities)
    else:
        return

    if len(companies) == 0:
        return

    preprocessed = preprocess_tweet(tweet)
    vectorized = transform_processed(preprocessed)

    data = json.dumps({'signature_name': 'serving_default', 'instances': vectorized})

    headers = {"content-type": "application/json"}

    sentiment_predictions = requests.post(PERSONAL_MODEL_URL, data=data, headers=headers)

    return sentiment_predictions


if __name__ == "__main__":
    res = entity_recognition('Beautiful Maine Lobsters will now move tariff-free to Europe! For first time in many '
                             'years. GREAT new deal by USTR levels playing field with Canada. Millions of $’s more in'
                             ' EXPORTS...')
    print(obtain_companies(res))
