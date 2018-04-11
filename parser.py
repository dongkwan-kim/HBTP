# -*- coding: utf-8 -*-

__author__ = 'Dongkwan Kim'


import os
from WriterWrapper import WriterWrapper


def tree_dir(twitter_year):
    return './rumor_detection_acl2017/{0}/tree/'.format(twitter_year)


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

    def __init__(self, parent_id, user_id, story_id, time_stamp):

        self.event_id = Event.event_id_counter
        Event.event_id_counter += 1

        self.parent_id = parent_id
        self.user_id = user_id
        self.story_id = story_id
        self.time_stamp = time_stamp


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
    writer = WriterWrapper('event_table')


if __name__ == '__main__':
    pass
