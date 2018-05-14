import numpy as np
from nltk import PorterStemmer
import pandas as pd
import re
from collections import Counter, defaultdict
import os
import pprint
import pickle
from copy import deepcopy
import random

DATA_PATH = '../data'
STORY_PATH = os.path.join(DATA_PATH, 'story', 'preprocessed-label')


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

    def __init__(self, story_path_list, stemmer=PorterStemmer, delimiter='\s', len_criteria=None, wf_criteria=None,
                 story_order='shuffle', force_save=False):
        """
        Attributes
        ----------
        :word_ids: list of list or dict of list
        idx -> [widx_0, widx_1, ..., widx_k]
        where k is len(set(words)) of story of idx,
              widx_z is a index of word in story of idx.

        :word_cnt: list of list or dict of list
        idx -> [cnt_0, cnt_1, ..., cnt_k]
        where k is len(set(words)) of story of idx,
              cnt_z is how many widx_z appeared in story of idx.

        :story_label: dict from id: int to label: str
        idx -> label (true, false, non-rumor, unverified)
        """
        self.story_path_list = story_path_list
        self.stemmer = stemmer()
        self.delimiter = delimiter
        self.len_criteria = len_criteria if len_criteria else lambda l: l > 1
        self.wf_criteria = wf_criteria if wf_criteria else lambda wf: 2 < wf < 500
        self.stop_words, self.stop_sentences = get_stops()
        self.force_save = force_save

        # Attributes that should be loaded
        self.story_order = story_order
        self.word_ids = None
        self.word_cnt = None
        self.story_label = None
        self.word_to_id = None
        self.id_to_word = None
        self.story_to_id = None

    def get_twitter_year(self):
        return 'twitter1516'

    def pprint(self):
        pprint.pprint(self.__dict__)

    def clone_with_only_mapping(self):
        tmp = deepcopy(self)
        tmp.word_ids = defaultdict(list)
        tmp.word_cnt = defaultdict(list)
        return tmp

    def remove_stop_sentences(self, content: str):
        for ss in self.stop_sentences:
            if ss in content:
                content = content.replace(ss, '')
        return content

    def clear_lambda(self):
        self.len_criteria = None
        self.wf_criteria = None

    def dump(self):
        file_name = 'FormattedStory_{}.pkl'.format(self.get_twitter_year())
        with open(os.path.join(STORY_PATH, file_name), 'wb') as f:
            self.clear_lambda()
            pickle.dump(self, f)
        print('Dumped: {0}'.format(file_name))

    def load(self):
        file_name = 'FormattedStory_{}.pkl'.format(self.get_twitter_year())
        try:
            with open(os.path.join(STORY_PATH, file_name), 'rb') as f:
                loaded = pickle.load(f)
                self.word_ids = loaded.word_ids
                self.word_cnt = loaded.word_cnt
                self.story_label = loaded.story_label
                self.word_to_id = loaded.word_to_id
                self.id_to_word = loaded.id_to_word
                self.story_to_id = loaded.story_to_id
                self.story_order = loaded.story_order
            print('Loaded: {0}'.format(file_name))
            return True
        except:
            print('Load Failed: {0}'.format(file_name))
            return False

    def get_formatted(self):

        if not self.force_save and self.load():
            return

        stories = pd.concat((pd.read_csv(path) for path in self.story_path_list), ignore_index=True)
        stories = stories.drop_duplicates(subset=['tweet_id'])
        stories = stories.reset_index(drop=True)

        # key: int, value: list
        story_id_to_contents = dict()

        # key: int, value: str
        story_id_to_label = dict()

        # key: str, value: int
        word_frequency = Counter()

        for i, story in stories.iterrows():
            content = story['title'] + '\n' + story['content']
            content = content.lower()
            content = self.remove_stop_sentences(content)

            words = re.split(self.delimiter, content)
            words = [self.stemmer.stem(v) for v in words]
            words = [v for v in words if v not in self.stop_words]
            words = [re.sub('[\W_]+', '', v) for v in words]
            words = [v for v in words if self.len_criteria(len(v))]

            word_frequency = sum((word_frequency, Counter(words)), Counter())

            tweet_id = str(story['tweet_id'])
            story_id_to_contents[tweet_id] = words

            label = str(story['label'])
            story_id_to_label[tweet_id] = label

            if i % 100 == 0 and __name__ == '__main__':
                print(i)

        story_list = list(set(story_id_to_contents.keys()))
        if self.story_order == 'shuffle':
            random.shuffle(story_list)
        elif self.story_order == 'sorted':
            story_list = sorted(story_list)
        else:
            raise NotImplementedError

        story_to_id = dict((story, idx) for idx, story in enumerate(story_list))
        id_to_contents = dict((story_to_id[story_id], contents)
                              for story_id, contents in story_id_to_contents.items())
        id_to_label = dict((story_to_id[story_id], label)
                           for story_id, label in story_id_to_label.items())

        # Cut by word_frequency
        for i in range(len(stories)):
            words_prev = id_to_contents[i]
            words = [word for word in words_prev if self.wf_criteria(word_frequency[word])]
            id_to_contents[i] = words

        # Construct a set of words
        vocab = set()
        for words in id_to_contents.values():
            vocab = vocab | set(words)

        word_to_id = {word: idx for idx, word in enumerate(sorted(vocab))}
        id_to_word = {idx: word for word, idx in word_to_id.items()}
        cid_to_wids = {i: [word_to_id[word] for word in id_to_contents[i]] for i in range(len(id_to_contents))}

        self.word_ids = [np.array(list(Counter(cid_to_wids[i]).keys())) for i in range(len(cid_to_wids))]
        self.word_cnt = [np.array(list(Counter(cid_to_wids[i]).values())) for i in range(len(cid_to_wids))]
        self.story_label = [id_to_label[i] for i in range(len(id_to_label))]
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.story_to_id = story_to_id

    def get_word_from_id(self, wid):
        if self.id_to_word:
            return self.id_to_word[wid]
        return None

    def get_id_from_word(self, word):
        if self.word_to_id:
            return self.word_to_id[word]
        return None


def get_formatted_stories(force_save=False) -> FormattedStory:
    fs = FormattedStory(get_story_files(), force_save=force_save)
    fs.get_formatted()
    return fs


if __name__ == '__main__':
    get_formatted_stories(force_save=True).dump()
