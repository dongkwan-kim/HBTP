# -*- coding: utf-8 -*-

__author__ = 'Dongkwan Kim'


import os
from WriterWrapper import WriterWrapper
from collections import defaultdict


OUTPUT_PATH = '../data/event/raw'


def tree_dir(twitter_year):
    return '../rumor_detection_acl2017/{0}/tree/'.format(twitter_year)


def get_tree_names(twitter_year):
    """
    :param twitter_year: str
    :return: list of filenames ['id.txt', ...]
    """
    t_dir = tree_dir(twitter_year)
    tree_names = os.listdir(t_dir)
    return tree_names


class Event:

    event_id_counter = 0
    event_list = defaultdict(list)

    def __init__(self, parent_id, user_id, story_id, time_stamp):

        self.event_id = Event.event_id_counter
        Event.event_id_counter += 1

        self.parent_id = parent_id
        self.user_id = user_id
        self.story_id = story_id
        self.time_stamp = time_stamp

        t = (parent_id, user_id, time_stamp)
        if t not in Event.event_list[story_id]:
            Event.event_list[story_id].append(t)
            self.is_unique = True
        else:
            self.is_unique = False

    def get_dict(self):
        d = self.__dict__
        del d['is_unique']
        return d


def event_one_line(line, story_id=None):
    """
    :param story_id:
    :param line: str "[str, str, str]->[str, str, str]"
    :return: Event
    """
    [parent, my_self] = [eval(x) for x in line.split('->')]
    parent_id = parent[0]
    user_id = my_self[0]

    if not story_id:
        story_id = parent[1]

    time_stamp = my_self[2]

    return Event(parent_id, user_id, story_id, time_stamp)


def event_table():
    """
    :return: .csv: [event_id, user_id, parent_id, story_id, time_stamp]
    """
    twitter_years = ['twitter15', 'twitter16']

    fieldnames = ['event_id', 'parent_id', 'user_id', 'story_id', 'time_stamp']
    for t_year in twitter_years:
        writer = WriterWrapper(os.path.join(OUTPUT_PATH, 'event_table_{0}'.format(t_year)), fieldnames)
        t_dir = tree_dir(t_year)
        t_tree_names = get_tree_names(t_year)
        for t_event_txt in t_tree_names:
            event_id = t_event_txt.split('.')[0]
            for line in open(os.path.join(t_dir, t_event_txt), 'r'):
                e = event_one_line(line, event_id)
                if e.is_unique:
                    writer.write_row(e.get_dict())
        Event.event_list = defaultdict(list)


if __name__ == '__main__':
    event_table()
