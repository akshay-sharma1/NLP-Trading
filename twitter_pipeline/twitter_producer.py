import tweepy
import os
import queue

import threading

# profile access tokens
ACCESS_TOKEN = os.environ.get('ACCESS_TOKEN')
ACCESS_TOKEN_SECRET = os.environ.get('ACCESS_TOKEN_SECRET')

# twitter_pipeline access tokens
TWITTER_CONSUMER_KEY = os.environ.get('TWITTER_CONSUMER_KEY')
TWITTER_CONSUMER_SECRET = os.environ.get('TWITTER_CONSUMER_SECRET')

# The Orange Man's ID
TRUMP_USER_ID = '25073877'

# Papa Elon's id
ELON_MUSK_USER_ID = '44196397'

ids = {TRUMP_USER_ID, ELON_MUSK_USER_ID}


class Twitter:
    def __init__(self):
        self.twitter_auth = tweepy.OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
        self.twitter_auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

        self.api = tweepy.API(auth_handler=self.twitter_auth,
                              retry_count=10,
                              retry_delay=1,
                              wait_on_rate_limit=True,
                              wait_on_rate_limit_notify=True)
        self.streamer = None

    def start_stream(self, sentiment_func):
        self.streamer = MyStreamListener(sentiment_func)
        my_stream = tweepy.Stream(auth=self.twitter_auth, listener=self.streamer)

        my_stream.filter([TRUMP_USER_ID, ELON_MUSK_USER_ID])

    def tweet_investment(self, companies, sentiment, link):
        # construct string with status
        status = ''
        if len(companies) == 1:
            status += companies[0] + ' is'
        else:
            status += ','.join(companies) + ' are'

        # handle all possible cases
        if not sentiment:
            status = ' no clue about ' + ','.join(companies) + emoji.emojize(' ')
        elif sentiment == 'positive':
            status += ' going to the Moon! Hop on the bull train now! 🐃'
        elif sentiment == 'negative':
            status += ' going in the toilet. Buy puts while you still can! 🐻'
        else:
            status += ' staying the same. Time to buy credit spreads i guess 😝'

        status += link
        self.api.update_status(status=status)


class MyStreamListener(tweepy.StreamListener):
    def __init__(self, sentiment_func):
        super().__init__()
        self.threads = []

        self.listening_queue = ListeningQueue(sentiment_func)
        self.start_threads()
        self.error_status = None

    def start_threads(self):
        for worker_id in range(10):
            worker = threading.Thread(target=self.listening_queue.process_queue)
            worker.daemon = True
            worker.start()
            self.threads.append(worker)

    def on_status(self, status):
        # load tweet status object
        try:
            tweet = status._json
        except ValueError:
            return

        # ensure that tweet is actually a tweet
        try:
            tweet_id = tweet['id_str']
            user_id = tweet['user']['id_str']
            name = tweet['user']['screen_name']
        except KeyError:
            return

        # Filter for either Trump or Elon's user ID
        if user_id not in ids:
            return

        meta_info = {}
        tweet_text = self.retrieve_text(status)

        meta_info['user_id'], meta_info['name'] = user_id, name

        meta_info['tweet'], meta_info['tweet_id'] = tweet_text, tweet_id

        print('sending data with name {}'.format(meta_info['name']))
        self.listening_queue.put_object(meta_info)
        return True

    def retrieve_text(self, status):

        # obtain the text from the tweet
        text = None
        if hasattr(status, "retweeted_status"):
            try:
                text = status.retweeted_status.extended_tweet["full_text"]
            except AttributeError:
                text = status.retweeted_status.text
        else:
            try:
                text = status.extended_tweet["full_text"]
            except AttributeError:
                text = status.text

        return text

    def on_error(self, status):
        self.error_status = status
        return False


class ListeningQueue:
    def __init__(self, sentiment_func):
        self.queue = queue.Queue()
        self.stop = threading.Event()

        self.sentiment_func = sentiment_func

    def put_object(self, obj):
        self.queue.put(obj)

    def process_queue(self):
        while not self.stop.is_set():
            try:
                tweet_info = self.queue.get(block=True, timeout=60)
                print('processing data with name {}'.format(tweet_info['name']))
                self.sentiment_func(tweet_info)
                self.queue.task_done()
            except queue.Empty:
                print('queue is empty..')
                continue
