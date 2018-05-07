import numpy as np
from nltk import PorterStemmer
import pandas as pd
import re
from collections import Counter
import os
import pprint


STORY_PATH = '../data/story/preprocessed'
STOP_WORDS = 'i, me, my, myself, we, our, ours, ourselves, you, your, yours, yourself, yourselves, he, him, his, himself, she, her, hers, herself, it, its, itself, they, them, their, theirs, themselves, what, which, who, whom, this, that, these, those, am, is, are, was, were, be, been, being, have, has, had, having, do, does, did, doing, a, an, the, and, but, if, or, because, as, until, while, of, at, by, for, with, about, against, between, into, through, during, before, after, above, below, to, from, up, down, in, out, on, off, over, under, again, further, then, once, here, there, when, where, why, how, all, any, both, each, few, more, most, other, some, such, no, nor, not, only, own, same, so, than, too, very, s, t, can, will, just, don, should, now'


def get_story_files():
    return [os.path.join(STORY_PATH, f) for f in os.listdir(STORY_PATH) if 'csv' in f]


class FormattedStory:

    def __init__(self, story_path, stemmer=PorterStemmer, delimiter='\s', len_criteria=None, wf_criteria=None):
        self.story_path = story_path
        self.stemmer = stemmer()
        self.delimiter = delimiter
        self.len_criteria = len_criteria if len_criteria else lambda l: l > 1
        self.wf_criteria = wf_criteria if wf_criteria else lambda wf: 2 < wf < 500

        self.word_ids = None
        self.word_cnt = None
        self.word_to_id = None
        self.id_to_word = None

    def pprint(self):
        pprint.pprint(self.__dict__)

    def get_formatted(self):
        stories = pd.read_csv(self.story_path)

        # key: int, value: list
        id_to_content = dict()

        # key: str, value: int
        word_frequency = Counter()

        for i in range(len(stories)):
            row = stories.ix[i]

            content = row['title'] + '\n' + row['content']
            content = content.lower()

            words = re.split(self.delimiter, content)
            words = [self.stemmer.stem(v) for v in words]
            words = [v for v in words if v not in STOP_WORDS]
            words = [re.sub('[\W_]+', '', v) for v in words]
            words = [v for v in words if self.len_criteria(len(v))]

            word_frequency = sum((word_frequency, Counter(words)), Counter())
            id_to_content[i] = words

            if i % 10 == 0 and __name__ == '__main__':
                print(i)

        # Cut by word_frequency
        for i in range(len(stories)):
            words_prev = id_to_content[i]
            words = [word for word in words_prev if self.wf_criteria(word_frequency[word])]
            id_to_content[i] = words

        # Construct a set of words
        vocab = set()
        for words in id_to_content.values():
            vocab = vocab | set(words)

        word_to_id = {word: idx for idx, word in enumerate(sorted(vocab))}
        id_to_word = {idx: word for word, idx in word_to_id.items()}
        cid_to_wids = {i: [word_to_id[word] for word in id_to_content[i]] for i in range(len(id_to_content))}

        self.word_ids = [np.array(list(Counter(cid_to_wids[i]).keys())) for i in range(len(cid_to_wids))]
        self.word_cnt = [np.array(list(Counter(cid_to_wids[i]).values())) for i in range(len(cid_to_wids))]
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word

    def get_word_from_id(self, wid):
        if self.id_to_word:
            return self.id_to_word[wid]
        return None

    def get_id_from_word(self, word):
        if self.word_to_id:
            return self.word_to_id[word]
        return None


def get_formatted_stories():
    r_list = []
    for story_path in get_story_files():
        fe = FormattedStory(story_path)
        fe.get_formatted()
        r_list.append(fe)
    return r_list


if __name__ == '__main__':
    for data in get_formatted_stories():
        data.pprint()
