import numpy as np
from nltk import PorterStemmer
import pandas as pd
import re
from collections import Counter
import os
import pprint
import pickle


DATA_PATH = '../data'
STORY_PATH = os.path.join(DATA_PATH, 'story', 'preprocessed')


def get_stops():
    """
    :return: tuple of two lists: ([...], [...])
    """
    stop_words = open(os.path.join(DATA_PATH, 'stopwords.txt'), 'r', encoding='utf-8').readlines()
    stop_sentences = open(os.path.join(DATA_PATH, 'stopsentences.txt'), 'r', encoding='utf-8').readlines()

    # strip, lower and reversed sort by sentence's length
    stop_sentences = sorted([ss.strip().lower() for ss in stop_sentences], key=lambda s: -len(s))
    stop_words = [sw.strip().lower() for sw in stop_words]

    return stop_words, stop_sentences


def get_story_files():
    return [os.path.join(STORY_PATH, f) for f in os.listdir(STORY_PATH) if 'csv' in f]


class FormattedStory:

    def __init__(self, story_path, stemmer=PorterStemmer, delimiter='\s', len_criteria=None, wf_criteria=None):
        self.story_path = story_path
        self.stemmer = stemmer()
        self.delimiter = delimiter
        self.len_criteria = len_criteria if len_criteria else lambda l: l > 1
        self.wf_criteria = wf_criteria if wf_criteria else lambda wf: 2 < wf < 500
        self.stop_words, self.stop_sentences = get_stops()

        self.word_ids = None
        self.word_cnt = None
        self.word_to_id = None
        self.id_to_word = None

    def get_twitter_year(self):
        return self.story_path.split('_')[2]

    def pprint(self):
        pprint.pprint(self.__dict__)

    def remove_stop_sentences(self, content: str):
        for ss in self.stop_sentences:
            if ss in content:
                content = content.replace(ss, '')
        return content

    def dump(self):
        file_name = 'FormattedStory_{}.pkl'.format(self.get_twitter_year())
        with open(os.path.join(STORY_PATH, file_name), 'wb') as f:
            pickle.dump(self, f)
        print('Dumped: {0}'.format(file_name))

    def load(self):
        file_name = 'FormattedStory_{}.pkl'.format(self.get_twitter_year())
        try:
            with open(os.path.join(STORY_PATH, file_name), 'rb') as f:
                loaded = pickle.load(f)
                self.word_ids = loaded.word_ids
                self.word_cnt = loaded.word_cnt
                self.word_to_id = loaded.word_to_id
                self.id_to_word = loaded.id_to_word
            print('Loaded: {0}'.format(file_name))
            return True
        except:
            print('Load Failed: {0}'.format(file_name))
            return False

    def get_formatted(self):

        if self.load():
            return

        stories = pd.read_csv(self.story_path)

        # key: int, value: list
        id_to_content = dict()

        # key: str, value: int
        word_frequency = Counter()

        for i in range(len(stories)):
            row = stories.loc[i]

            content = row['title'] + '\n' + row['content']
            content = content.lower()
            content = self.remove_stop_sentences(content)

            words = re.split(self.delimiter, content)
            words = [self.stemmer.stem(v) for v in words]
            words = [v for v in words if v not in self.stop_words]
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
        data.dump()
