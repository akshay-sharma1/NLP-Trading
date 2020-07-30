import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
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
    print(res)

    ax = sns.heatmap(res)
    ax.show()


def sentiment_over_time(data):
    pass


def main():
    preprocessed = read_labeled_data()
    mood_distribution(preprocessed)
    n_most_common(preprocessed, 20)


if __name__ == '__main__':
    main()
