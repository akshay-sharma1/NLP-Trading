from twitter_producer import Twitter
from twitter_consumer import entity_recognition, obtain_companies, predict_sentiment, category_recognition


def get_orig_tweet_link(tweet_info):
    BASE = 'http://twitter.com'
    return '/'.join([BASE, tweet_info['name'].replace(' ', ''), 'status', tweet_info['tweet_id']])


def analyze_and_tweet(tweet_info):
    text = tweet_info['tweet']

    # recognize entities and companies
    entities = entity_recognition(text)
    companies = obtain_companies(entities=entities)

    econ_index = category_recognition(text)

    # if tweet isn't about a company or the economy, stop
    if not companies and not econ_index:
        return
    else:
        if not companies:
            companies = econ_index

    sentiment = predict_sentiment(text)
    link = get_orig_tweet_link(tweet_info)
    twitter_handler = Twitter()

    twitter_handler.tweet_investment(companies=companies, sentiment=sentiment, link=link)


def main():
    api = Twitter()
    api.start_stream(analyze_and_tweet)


if __name__ == "__main__":
    main()
