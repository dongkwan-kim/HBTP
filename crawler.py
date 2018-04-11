# -*- coding: utf-8 -*-

__author__ = 'Dongkwan Kim'


import twitter
import configparser
from newspaper import Article
from WriterWrapper import WriterWrapper
import time


def label_path(twitter_year):
    return './rumor_detection_acl2017/{0}/label.txt'.format(twitter_year)


def get_id_label_list(path):
    """
    :param path: Result of get_label_path
    :return: [{'twee_id': str, 'label': str}, ...]
    """
    label_txt = open(path, "r").readlines()

    # [[label, tweet_id], ...]
    label_pairs = [x.strip().split(':') for x in label_txt]

    return [{
        'tweet_id': tweet_id,
        'label': label
    } for label, tweet_id in label_pairs]


def api_twitter(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)
    config_t = config['TWITTER']

    consumer_key = config_t['CONSUMER_KEY']
    consumer_secret = config_t['CONSUMER_SECRET']
    access_token = config_t['ACCESS_TOKEN']
    access_token_secret = config_t['ACCESS_TOKEN_SECRET']

    _api = twitter.Api(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token_key=access_token,
        access_token_secret=access_token_secret
    )

    return _api


class TwitterAPIWrapper:

    def __init__(self, api_path):
        self.api = api_twitter(api_path)

    def get_status(self, status_id):
        status = self.api.GetStatus(status_id)
        return status

    def get_status_dict(self, status_id):
        status = self.get_status(status_id)
        return status.AsDict()

    def get_what_we_want(self, status_id, delay=1):
        """
        :param delay: int or float (sec)
        :param status_id:
        :return: {urls: list of str, text: str}
        """
        status_dict = self.get_status_dict(status_id)

        # Timer delay
        time.sleep(delay)

        # urls: list
        urls = status_dict['urls']
        urls_list = [x['expanded_url'] for x in urls]

        # text: str
        text = status_dict['text']

        return {
            'urls': urls_list,
            'tweet_text': text,
        }

    # www = what we want
    def get_www_flatten(self, status_id, delay=1):
        """
        :param delay: int or float (sec)
        :param status_id: str, ...
        :return: [{'url': str, 'text': str}, ...]
        """
        r = []
        try:
            www = self.get_what_we_want(status_id, delay)
        except Exception as e:
            return [{'url': str(e), 'tweet_text': ''}]

        www_text = www['tweet_text']
        for url in www['urls']:
            r.append({
                'url': url,
                'tweet_text': www_text
            })
        return r


def get_contents(url):
    """
    :param url:
    :return: {'crawled_or_error_log': boolean or str, 'title': str, 'content': str]
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        r = {
            'crawled_or_error_log': True,
            'title': article.title,
            'content': article.text
        }
    except Exception as e:
        r = {
            'crawled_or_error_log': str(e),
            'title': '',
            'content': ''
        }

    return r


def merge_dicts(lst_of_dct):
    new_dict = {}
    for dct in lst_of_dct:
        new_dict.update(dct)
    return new_dict


def story_table(config_name):
    my_api = TwitterAPIWrapper(config_name)

    fieldnames = ['tweet_id', 'label', 'tweet_text', 'url', 'crawled_or_error_log', 'title', 'content']

    twitter_years = ['twitter16', 'twitter15']

    # str
    for ty in twitter_years:
        writer = WriterWrapper('story_table_{0}'.format(ty), fieldnames)
        id_label_list = get_id_label_list(label_path(ty))

        # {'tweet_id': str, 'label': str}
        for id_label in id_label_list:
            tweet_id = id_label['tweet_id']
            www_list = my_api.get_www_flatten(tweet_id)

            # {'url': str, 'tweet_text': str}
            for www in www_list:
                url = www['url']
                content_dict = get_contents(url)

                merged_dict = merge_dicts([content_dict, www, id_label])
                writer.write_row(merged_dict)
                print(merged_dict)


if __name__ == '__main__':
    story_table('./config.ini')
