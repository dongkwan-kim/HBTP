# -*- coding: utf-8 -*-

__author__ = 'Dongkwan Kim'


import os
import csv
from WriterWrapper import WriterWrapper


DATA_PATH = '../data'
STORY_PATH = os.path.join(DATA_PATH, 'story', 'preprocessed')
EVENT_PATH = os.path.join(DATA_PATH, 'event', 'raw')
OUTPUT_PATH = os.path.join(DATA_PATH, 'event', 'synchronized')


def synchronize_event():
    events = [e for e in os.listdir(EVENT_PATH) if 'csv' in e]
    stories = [s for s in os.listdir(STORY_PATH) if 'csv' in s]

    for ef_name, sf_name in zip(events, stories):
        ef = open(os.path.join(EVENT_PATH, ef_name), 'r', encoding='utf-8')
        sf = open(os.path.join(STORY_PATH, sf_name), 'r', encoding='utf-8')

        s_reader = csv.DictReader(sf)
        story_id_list = [line['tweet_id'] for line in s_reader]

        e_reader = csv.DictReader(ef)

        writer_file = os.path.join(OUTPUT_PATH, '_'.join(ef_name.split('_')[:-1]))
        writer = WriterWrapper(writer_file, e_reader.fieldnames)

        for line in e_reader:
            if line['story_id'] in story_id_list:
                writer.write_row(line)


if __name__ == '__main__':
    synchronize_event()
