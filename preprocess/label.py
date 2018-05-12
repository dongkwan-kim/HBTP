# -*- coding: utf-8 -*-

__author__ = 'Dongkwan Kim'

import os
import csv
from WriterWrapper import WriterWrapper
from collections import defaultdict
from pprint import pprint


DATA_PATH = '../data'
STORY_PATH = os.path.join(DATA_PATH, 'story')
INPUT_PATH = os.path.join(STORY_PATH, 'preprocessed')
OUTPUT_PATH = os.path.join(STORY_PATH, 'preprocessed-label')


def get_title_to_multiple_label_tuple() -> dict:
    """
    :return: title:str -> list of tuple (true | false | non-rumor | unverified)
    """
    csv_files = [f for f in os.listdir(INPUT_PATH) if 'csv' in f]

    title_to_tid_and_label = defaultdict(set)
    for csv_f in csv_files:
        f = open(os.path.join(INPUT_PATH, csv_f), 'r', encoding='utf-8')
        reader = csv.DictReader(f)

        for line in reader:
            tweet_id, label, title = line['tweet_id'], line['label'], line['title']
            title_to_tid_and_label[title].add(label)

    multiple_label_story = dict((title, tuple(sorted(label)))
                                for title, label in title_to_tid_and_label.items() if len(label) != 1)
    pprint(multiple_label_story)
    return multiple_label_story


def preprocess_label():
    csv_files = [f for f in os.listdir(INPUT_PATH) if 'csv' in f]

    title_to_multiple_label_tuple = get_title_to_multiple_label_tuple()

    for csv_f in csv_files:
        f = open(os.path.join(INPUT_PATH, csv_f), 'r', encoding='utf-8')
        reader = csv.DictReader(f)
        writer_file = os.path.join(OUTPUT_PATH, '_'.join(csv_f.split('_')[:-1]))
        writer = WriterWrapper(writer_file, reader.fieldnames)

        for line in reader:
            title = line['title']
            if title in title_to_multiple_label_tuple:
                label_tuple = title_to_multiple_label_tuple[line['title']]
                # This is the only case for multiple unique labels.
                if label_tuple == ('true', 'unverified'):
                    line['label'] = 'true'

            writer.write_row(line)

        f.close()


if __name__ == '__main__':
    preprocess_label()
