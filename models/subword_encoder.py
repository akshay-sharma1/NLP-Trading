import re

from collections import defaultdict


def get_pairs(word):
    """
        Helper function that gets all unique pairs of characters in a word
        using two-pointers and a set.

            args:
                word: str
            ret:
                pairs: set that has all consecutive unique character pairs

    """
    pairs = set()

    # returns all unique pairs using two-pointers
    slow = word[0]
    for i in range(1, len(word)):
        pairs.add((slow, word[i]))
        slow = word[i]

    return pairs


class BPE:
    """
        Helper class that encodes a word based on its subwords through a
        series of consecutive merges.
    """
    def __init__(self, data, num_merges):
        # core data structures
        self.vocab = defaultdict(int)
        self.bpe_hash = {}

        self.num_merges = num_merges
        self.data = data

    def create_vocab(self):
        """
            Function that creates the initial vocab at the individual word level.
            Splits each word into individual characters with a space between them and
            appends end of word character (<w>). Modifies class vocab attribute by
            creating mappings between words and their frequencies.

        """
        for sentence in self.data:
            for word in sentence.split():
                self.vocab[' '.join(list(word)) + ' <w>'] += 1

    def get_pair_freqs(self):
        """
            Calculates character-level frequencies of each word in the vocabulary
            on each iteration. Is used to calculate the 'best' possible candidate for a
            pair-wise merge based on frequencies. (e.g if (e r) have the highest frequency,
            merge them to form (er) in the new vocab as a separate token).

            ret:
                pair_stats: frequencies of individual character pairs in vocabulary

        """
        pair_stats = defaultdict(int)

        for word, freq in self.vocab.items():
            chars = word.split()
            for i in range(len(chars) - 1):
                pair_stats[chars[i], chars[i + 1]] += freq

        return pair_stats

    def merge_vocab(self, pair):
        """
           Merges the most frequently occurring pair of characters in each iteration.
           Creates new vocab with newly encoded token replacing the old one, and sets
           the class vocab to the new one.

        """
        # merges 'best' unigram pair into a bigram
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

        # generate new vocab dictionary based on merged pair
        for word in self.vocab:
            w_out = p.sub(''.join(pair), word)
            new_vocab[w_out] = self.vocab[word]

        self.vocab = new_vocab

    def get_token_freqs(self):
        # returns individual characters or n-grams (total data tokens)
        # after each merge
        token_freqs = defaultdict(int)

        for word, freq in self.vocab.items():
            for token in word.split():
                token_freqs[token] += freq

        # obtain sorted tokens in terms of length
        sorted_tokens = list(token_freqs.keys())
        sorted_tokens.sort(key=lambda x: len(x[:-4]) + 1 if x[-4:] == '<w>' else len(x), reverse=True)

        return token_freqs, sorted_tokens

    def train_vocab(self):
        """
            Comprehensive routine that trains the subword_encoder on a corpus of tweets.
            Initially creates a vocab using the create_vocab class subroutine, iterates
            for num_merges, calculates and merges most frequently occurring pair on each merge,
            and saves the best pair token into a hashmap (self.bpe_hash) that is used to encode
            new words in the encode_words routine.

        """

        self.create_vocab()
        # merge vocab for num_merges iterations
        for i in range(self.num_merges):
            pair_stats = self.get_pair_freqs()
            if not pair_stats:
                break

            best = max(pair_stats, key=pair_stats.get)
            self.merge_vocab(best)

            self.bpe_hash[best] = i

    def encode_word(self, word):
        """
            Uses fitted subword_encoder (BPE instantiation) to encode a new word based
            on fitted vocab tokens. Initially, gets all character pairs in the word, and
            encodes the word based on all accumulated subwords in the vocab.

            For example, if we had 'highest' as our input, and our vocabulary had the following
            tokens: ['hi', 'ghe', 'st', 'mount', 'ain'], 'highest' would be encoded as:

            'highest' => ['hi', 'ghe', 'st']

            based on our accumulated vocab tokens
        """
        word = list(word)
        word.append('<w>')

        pairs = get_pairs(word)
        if not pairs:
            return word

        # iterate until particular condition is reached
        iteration = 0
        while True:
            iteration += 1

            # get the bigram we saved in the hashmap during merges
            bigram = min(pairs, key=lambda pair: self.bpe_hash.get(pair, float('inf')))
            if bigram not in self.bpe_hash:
                break

            first, second = bigram
            new_word = []

            # attempt to construct the word based on where the
            # bigram is in the word
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i + 1 < len(word) and word[i + 1] == second:
                    new_word.append(bigram[0] + bigram[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        # remove end-of-word symbols
        if word[-1] == '<w>':
            word = word[:-1]
        elif word[-1].endswith('<w>'):
            word[-1] = word[-1].replace('<w>', '')

        return word
