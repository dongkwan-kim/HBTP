# -*- coding: utf-8 -*-

__author__ = 'Dongkwan Kim'


import os


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


if __name__ == '__main__':
    print(get_tree_names('twitter16'))
