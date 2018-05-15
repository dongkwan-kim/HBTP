# -*- coding: utf-8 -*-

__author__ = 'Dongkwan Kim'


from TwitterAPIWrapper import TwitterAPIWrapper
from collections import defaultdict
import os
import time
import pickle
import pandas as pd


DATA_PATH = '../data'
EVENT_PATH = os.path.join(DATA_PATH, 'event', 'synchronized')
NETWORK_PATH = os.path.join(DATA_PATH, 'network')


def get_event_files():
    return [os.path.join(EVENT_PATH, f) for f in os.listdir(EVENT_PATH) if 'csv' in f]


class UserNetwork:

    def __init__(self, user_id_to_follower_ids=None, user_id_to_friend_ids=None):
        """
        :param user_id_to_follower_ids: collection of user IDs for every user following the key-user.
        :param user_id_to_friend_ids: collection of user IDs for every user the key-user is following.
        """
        self.user_id_to_follower_ids = user_id_to_follower_ids
        self.user_id_to_friend_ids = user_id_to_friend_ids

    def dump(self):
        file_name = 'UserNetwork.pkl'
        with open(os.path.join(NETWORK_PATH, file_name), 'wb') as f:
            pickle.dump(self, f)
        print('Dumped: {0}'.format(file_name))

    def load(self):
        file_name = 'UserNetwork.pkl'
        try:
            with open(os.path.join(NETWORK_PATH, file_name), 'rb') as f:
                loaded = pickle.load(f)
                self.user_id_to_follower_ids = loaded.user_id_to_follower_ids
                self.user_id_to_friend_ids = loaded.user_id_to_friend_ids
            print('Loaded: {0}'.format(file_name))
            return True
        except:
            print('Load Failed: {0}.\n'.format(file_name),
                  'If you want to get UserNetwork, please refer UserNetworkAPIWrapper')
            return False

    def get_follower_ids(self, user_id):
        return self.user_id_to_follower_ids[user_id]

    def get_friend_ids(self, user_id):
        return self.user_id_to_friend_ids[user_id]


class UserNetworkAPIWrapper(TwitterAPIWrapper):

    def __init__(self, config_file_path, event_path_list):
        """
        Attributes
        ----------
        :user_id_to_follower_ids: dict, str -> list
        :user_id_to_friend_ids: dict, str -> list
        """
        super().__init__(config_file_path)
        self.event_path_list = event_path_list
        self.user_id_to_follower_ids = dict()
        self.user_id_to_friend_ids = dict()

    def dump_user_network(self):
        user_network = UserNetwork(
            self.user_id_to_follower_ids,
            self.user_id_to_friend_ids,
        )
        user_network.dump()
        return user_network

    def get_user_network(self):
        print('Just called get_user_network(), which is a really heavy method.')

        user_set = self.get_user_id_full_set()
        user_to_id = dict((user, idx) for idx, user in enumerate(sorted(user_set)))

        # user_id: str
        for user_id in user_set:

            if user_id == 'ROOT':
                continue

            print(user_id)
            self.user_id_to_follower_ids[user_id] = self._fetch_follower_ids(user_id)

        self.user_id_to_follower_ids = self.indexify(self.user_id_to_follower_ids, user_to_id, user_to_id)

        return self.dump_user_network()

    def get_user_id_full_set(self):
        return self.get_user_id_set()

    def get_user_id_set(self, length=None) -> set:
        events = self.get_events(self.event_path_list)

        parent_to_child = defaultdict(list)
        user_to_stories = defaultdict(list)

        user_set = set()

        # Construct a dict from feature to feature
        for i, event in events.iterrows():
            parent, user, story = map(str, [event['parent_id'], event['user_id'], event['story_id']])

            parent_to_child[parent].append(user)
            user_to_stories[user].append(story)
            user_set.update([parent, user])

            if i % 10000 == 0 and __name__ == '__main__':
                print(i)

            # This is for the test.
            if length is not None and i > length:
                break

        leaf_users = self.get_leaf_user_set(parent_to_child, user_to_stories)
        user_set = set(u for u in user_set if u not in leaf_users)
        return user_set

    def get_events(self, event_path_list):
        events = pd.concat((pd.read_csv(path) for path in event_path_list), ignore_index=True)

        # Remove duplicated events
        events = events.drop(['event_id'], axis=1)
        events = events.drop_duplicates()
        events = events.reset_index(drop=True)

        return events

    def get_leaf_user_set(self, parent_to_child, user_to_stories):
        leaf_users = set()
        for parent, child in parent_to_child.items():
            for leaf_user in child:
                if leaf_user not in parent_to_child and len(user_to_stories[leaf_user]) == 1:
                    leaf_users.add(leaf_user)
        return leaf_users

    def indexify(self, target_dict: dict, key_to_id: dict, value_to_id: dict):
        """
        :param target_dict: dict {key -> list of values}
        :param key_to_id: dict
        :param value_to_id: dict
        :return: dict {key_to_id[key] -> value_to_id[value]}
        """
        r_dict = {}
        for key, values in target_dict.items():
            r_dict[key_to_id[key]] = list(map(lambda v: value_to_id[v], values))
        return r_dict

    def paged_to_all(self, user_id, paged_func):

        all_list = []

        while True:
            prev_cursor, next_cursor, partial_list = paged_func(user_id)
            all_list += partial_list
            print('Fetched {0} of {1}, Sleep(60s), Next cursor is {2}'.format(
                len(partial_list), paged_func.__name__, next_cursor
            ))

            if next_cursor == 0 or next_cursor == prev_cursor:
                break
            else:
                time.sleep(60)

        return all_list

    def _fetch_follower_ids(self, user_id) -> list:
        return self.paged_to_all(user_id, self._fetch_follower_ids_paged)

    def _fetch_friend_ids(self, user_id) -> list:
        return self.paged_to_all(user_id, self._fetch_friend_ids_paged)

    def _fetch_follower_ids_paged(self, user_id, cursor=-1) -> (int, int, list):
        # TODO: Implement this, with self.api.GetFollowerIDsPaged()
        # http://python-twitter.readthedocs.io/en/latest/twitter.html#twitter.api.Api.GetFollowerIDsPaged
        return 0, 0, []

    def _fetch_friend_ids_paged(self, user_id, cursor=-1) -> (int, int, list):
        # TODO: Implement this, with self.api.GetFriendIDsPaged()
        # http://python-twitter.readthedocs.io/en/latest/twitter.html#twitter.api.Api.GetFriendIDsPaged
        return 0, 0, []


if __name__ == '__main__':
    user_network_api = UserNetworkAPIWrapper('./config.ini', get_event_files())
    user_network_api.get_user_network()
