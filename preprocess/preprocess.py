# -*- coding: utf-8 -*-

__author__ = 'Dongkwan Kim'


import os
import csv
import re
import requests
from WriterWrapper import WriterWrapper
from urllib.parse import urlparse


DATA_PATH = '../data'
STORY_PATH = os.path.join(DATA_PATH, 'story')
INPUT_PATH = os.path.join(STORY_PATH, 'implicit-error-preprocessed')
OUTPUT_PATH = os.path.join(STORY_PATH, 'preprocessed')


def get_stopwords():
    stopwords = open(os.path.join(DATA_PATH, 'stopwords.txt'), 'r', encoding='utf-8').readlines()
    return [sw.lower().strip() for sw in stopwords]


def contain_stopwords(text, stopwords):
    text = text.lower()
    for sw in stopwords:
        # if sw in text.split(' '):
        if sw == text:
            return True
    return False


def is_link_expired(url):
    try:
        r = requests.get(url)
        parsed = urlparse(r.url)
        path = parsed[2]
        return path == '/'
    except:
        print(url)
        return True


def preprocess_story():
    csv_files = [f for f in os.listdir(INPUT_PATH) if 'csv' in f]
    stopwords = get_stopwords()

    for csv_f in csv_files:
        f = open(os.path.join(INPUT_PATH, csv_f), 'r')
        reader = csv.DictReader(f)
        writer_file = os.path.join(OUTPUT_PATH, '_'.join(csv_f.split('_')[:-1]))
        writer = WriterWrapper(writer_file, reader.fieldnames)

        for line in reader:

            url = line['url']
            if is_link_expired(url):
                print('Expired {0}'.format(url))
                continue

            content = re.sub('\n+', '\n', line['content'])
            content = content.split('\n')

            for c in content:
                if contain_stopwords(c, stopwords):
                    content.remove(c)

            content = '\n'.join(content)
            line['content'] = content

            writer.write_row(line)

        f.close()


if __name__ == '__main__':
    preprocess_story()

