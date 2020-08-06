import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from wordcloud import WordCloud
from PIL import Image
from preprocess import process_labeled_data


def read_labeled_data():
    data = pd.read_csv('./data/labeled_data.csv')

    return process_labeled_data(data)


def mood_distribution(data):
    mood_dataframe = data['sentiment'].value_counts()

    plt.bar([1, 2, 3], mood_dataframe, color=['r', 'b', 'g'])
    plt.xticks([1, 2, 3], ['Negative', 'Positive', 'Neutral'])
    plt.xlabel('Mood')
    plt.ylabel('# Tweets')
    plt.title('Mood Distribution')

    plt.show()


def n_most_common(data, n):
    top_words = Counter([word for sublist in data['text'] for word in sublist])
    res = pd.DataFrame(top_words.most_common(n), columns=['Word', 'count'])
    res.style.background_gradient(cmap='PuBu')
    return res


def sentiment_over_time(data):
    # standardize datetime by converting to DateTimeIndex
    data['created_at'] = pd.to_datetime(data['created_at'])

    # remove all rows that are neutral
    pos_neg = data[data['sentiment'] != 1]

    # group sentiment by month
    monthly_sentiment = pos_neg.groupby([pd.Grouper(key='created_at', freq='1M'), 'sentiment']).size().\
        unstack('sentiment').fillna(0)

    # plot graph
    monthly_sentiment.plot.line()
    plt.title('Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('# Tweets')
    plt.legend(labels=['Negative', 'Positive'])

    plt.show()


def generate_word_cloud(data):
    # DeNest tweets array, and then join it into a string of words separated by comma
    words = ' '.join([' '.join(tweet) for tweet in data['text']])

    # load image and create worldcloud
    twitter_mask = np.array(Image.open('/Users/aksharma/PycharmProjects/Trump_Tweets/twitter_mask.png'))
    wec = WordCloud(background_color='white', max_words=1000, mask=twitter_mask, width=1800, height=1400,
                    contour_color='steelblue', contour_width=3)
    wec.generate(words)

    plt.imshow(wec, interpolation='bilinear')
    plt.title('Twitter Word Cloud')
    plt.axis('off')
    plt.show()


def visualize_word_distance(vectorized_data):
    pass


def main():
    preprocessed = read_labeled_data()

    mood_distribution(preprocessed)

    print(n_most_common(preprocessed, 20))
    sentiment_over_time(preprocessed)

    generate_word_cloud(preprocessed)


if __name__ == '__main__':
    main()
