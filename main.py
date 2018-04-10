# -*- coding: utf-8 -*-

__author__ = 'Dongkwan Kim'


import twitter
import configparser


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

    def get_what_we_want(self, status_id):
        """
        :param status_id:
        :return: {urls: -, text: -}
        """
        status_dict = self.get_status_dict(status_id)

        # urls: list
        urls = status_dict['urls']
        urls_list = [x['expanded_url'] for x in urls]

        # text: str
        text = status_dict['text']

        return {
            'urls': urls_list,
            'text': text,
        }


if __name__ == '__main__':
    my_api = TwitterAPIWrapper('./config.ini')
