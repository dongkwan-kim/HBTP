# -*- coding: utf-8 -*-

__author__ = 'Dongkwan Kim'

import os
import csv
from pprint import pprint
from collections import Counter
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

DATA_PATH = '../data'
STORY_PATH = os.path.join(DATA_PATH, 'story', 'preprocessed')
EVENT_PATH = os.path.join(DATA_PATH, 'event', 'synchronized')


def build_hist(enumerable, title, twitter_year, config):
    _bins = 50 if 'bins' not in config else config['bins']
    _range = None if 'range' not in config else config['range']

    n, bins, patches = plt.hist(
        list(enumerable),
        bins=_bins,
        range=_range,
    )
    plt.title('{0} - {1}'.format(title, twitter_year))
    plt.grid(True)
    plt.show()

    return n, bins, patches


def story_stats():
    """
    csv: ['tweet_id', 'label', 'tweet_text', 'url', 'crawled_or_error_log', 'title', 'content', 'Content size']
    :return:
    """
    stories = [os.path.join(STORY_PATH, f) for f in os.listdir(STORY_PATH) if 'csv' in f]
    r_dict = {}

    for sf in stories:
        f = open(sf, 'r', encoding='utf-8')
        reader = csv.DictReader(f)

        size = 0
        for line in reader:
            size += 1

        stats = {
            'size': size,
        }

        twitter_year = sf.split('_')[2]
        r_dict[twitter_year] = stats

    return r_dict


def get_depth(event_dict_by_user_id, target_user_id):
    # find parent_node, then recursively call this function.
    event = event_dict_by_user_id[target_user_id]
    parent_id = event['parent_id']

    print(target_user_id)
    if parent_id == 'ROOT':
        return 0
    else:
        return 1 + get_depth(event_dict_by_user_id, parent_id)


def event_stats():
    """
    csv: ['event_id', 'parent_id', 'user_id', 'story_id', 'time_stamp']
    :return:
    """
    events = [os.path.join(EVENT_PATH, f) for f in os.listdir(EVENT_PATH) if 'csv' in f]
    r_dict = {}

    for ef in events:
        f = open(ef, 'r', encoding='utf-8')
        twitter_year = ef.split('_')[2]
        reader = csv.DictReader(f)

        event_list = list(reader)
        event_dict_by_user_id = dict((e['user_id'], e) for e in event_list)

        user_size = len(set([e['user_id'] for e in event_list]))
        users_per_story = Counter([e['user_id'] for e in event_list]).values()
        stories_per_user = Counter(e['story_id'] for e in event_list).values()
        for e in event_list:
            if '19937350' == e['user_id']:
                print(e)
            if '617825073' == e['user_id']:
                print(e)

        # depth = Counter([get_depth(event_dict_by_user_id, e['user_id']) for e in event_list])

        """
        n, bins, patches = build_hist(users_per_story, 'users/story', twitter_year, {
            'range': [0, 10],
        })

        n, bins, patches = build_hist(stories_per_user, 'stories/user', twitter_year, {})

        n, bins, patches = build_hist(depth, 'depth', twitter_year, {})

        """

if __name__ == '__main__':
    event_stats()
