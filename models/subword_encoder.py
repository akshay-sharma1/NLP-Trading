import re

from collections import defaultdict


def get_pairs(word):
    pairs = set()

    # returns all unique pairs using two-pointers
    slow = word[0]
    for i in range(1, len(word)):
        pairs.add((slow, word[i]))
        slow = word[i]

    return pairs


class BPE:
    def __init__(self, data, num_merges):
        # core data structures
        self.vocab = defaultdict(int)
        self.bpe_hash = {}

        self.num_merges = num_merges
        self.data = data

    def create_vocab(self):
        for sentence in self.data:
            for word in sentence.split():
                self.vocab[' '.join(list(word)) + ' <w>'] += 1

    def get_pair_freqs(self):
        pair_stats = defaultdict(int)

        for word, freq in self.vocab.items():
            chars = word.split()
            for i in range(len(chars) - 1):
                pair_stats[chars[i], chars[i + 1]] += freq

        return pair_stats

    def merge_vocab(self, pair):
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
        word = list(word)
        word.append('<w>')

        pairs = get_pairs(word)
        if not pairs:
            return word

        iteration = 0
        while True:
            iteration += 1

            bigram = min(pairs, key=lambda pair: self.bpe_hash.get(pair, float('inf')))
            if bigram not in self.bpe_hash:
                break

            first, second = bigram
            new_word = []

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
