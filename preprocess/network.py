# -*- coding: utf-8 -*-

__author__ = 'Dongkwan Kim'


from TwitterAPIWrapper import TwitterAPIWrapper
from collections import defaultdict
import os
import time
import pickle
import pandas as pd
from tqdm import tqdm


DATA_PATH = '../data'
EVENT_PATH = os.path.join(DATA_PATH, 'event', 'synchronized')
NETWORK_PATH = os.path.join(DATA_PATH, 'network')


def wait_second(sec=60):
    time.sleep(1)
    for _ in tqdm(range(sec)):
        time.sleep(1)


def get_event_files():
    return [os.path.join(EVENT_PATH, f) for f in os.listdir(EVENT_PATH) if 'csv' in f]


class UserNetwork:

    def __init__(self, user_id_to_follower_ids=None, user_id_to_friend_ids=None, user_set=None):
        """
        :param user_id_to_follower_ids: collection of user IDs for every user following the key-user.
        :param user_id_to_friend_ids: collection of user IDs for every user the key-user is following.
        """
        self.user_id_to_follower_ids = user_id_to_follower_ids
        self.user_id_to_friend_ids = user_id_to_friend_ids
        self.user_set = user_set

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
                self.user_set = loaded.user_set
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

    def __init__(self, config_file_path, event_path_list, story_ids: list=None, inspect_length: int=None):
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
        self.user_set = self.get_user_id_set_of_stories(story_ids, inspect_length)

        print('UserNetworkAPI initialized with {0} users.'.format(len(self.user_set)))

    def dump_user_network(self):
        user_network_for_dumping = UserNetwork(
            self.user_id_to_follower_ids,
            self.user_id_to_friend_ids,
            self.user_set,
        )
        user_network_for_dumping.dump()
        return user_network_for_dumping

    def get_user_network(self):
        time_to_wait = 5
        print('Just called get_user_network(), which is a really heavy method.\n',
              'This will start after {0}s.'.format(time_to_wait))
        wait_second(time_to_wait)

        # We are not using self.get_user_id_to_follower_ids() for now.
        self.get_user_id_to_friend_ids()

        time.sleep(1)
        return self.dump_user_network()

    def get_user_id_to_follower_ids(self):
        # user_id: str
        for user_id in self.user_set:
            if user_id != 'ROOT':
                self.user_id_to_follower_ids[user_id] = self._fetch_follower_ids(user_id)

    def get_user_id_to_friend_ids(self):
        # user_id: str
        for user_id in self.user_set:
            if user_id != 'ROOT':
                self.user_id_to_friend_ids[user_id] = self._fetch_friend_ids(user_id)

    def get_user_id_set_of_stories(self, story_ids: list=None, inspect_length: int=None) -> set:
        """
        :param story_ids: list of str or None
        :param inspect_length: int or None
        :return: set of user_id:int
        """

        if story_ids:
            story_ids = list(map(str, story_ids))

        events = self.get_events(self.event_path_list)

        parent_to_child = defaultdict(list)
        user_to_stories = defaultdict(list)

        user_set = set()

        # Construct a dict from feature to feature
        for i, event in events.iterrows():
            parent, user, story = map(str, [event['parent_id'], event['user_id'], event['story_id']])

            if story_ids is None or story in story_ids:
                parent_to_child[parent].append(user)
                user_to_stories[user].append(story)
                user_set.update([parent, user])

            if i % 10000 == 0 and __name__ == '__main__':
                print(i)

            # This is for the test.
            if inspect_length is not None and i > inspect_length:
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

    def paged_to_all(self, user_id, paged_func):

        all_list = []
        sec_to_wait = 60
        next_cursor = -1

        while True:
            next_cursor, prev_cursor, partial_list = paged_func(user_id, next_cursor)
            all_list += partial_list

            fetch_stop = next_cursor == 0 or next_cursor == prev_cursor
            print('Fetched user({0})\'s {1} of {2}, Stopped: {3}'.format(
                user_id, len(all_list), paged_func.__name__, fetch_stop
            ))
            wait_second(sec_to_wait)

            if fetch_stop:
                break

        return all_list

    def _fetch_follower_ids(self, user_id) -> list:
        try:
            return self.paged_to_all(user_id, self._fetch_follower_ids_paged)
        except Exception as e:
            print('Error in follower ids: {0}'.format(user_id), e)
            return []

    def _fetch_friend_ids(self, user_id) -> list:
        try:
            return self.paged_to_all(user_id, self._fetch_friend_ids_paged)
        except Exception as e:
            print('Error in friend ids: {0}'.format(user_id), e)
            return []

    def _fetch_follower_ids_paged(self, user_id, cursor=-1) -> (int, int, list):
        # http://python-twitter.readthedocs.io/en/latest/twitter.html#twitter.api.Api.GetFollowerIDsPaged
        next_cursor, prev_cursor, follower_ids = self.api.GetFollowerIDsPaged(
            user_id=user_id,
            cursor=cursor,
        )
        return next_cursor, prev_cursor, follower_ids

    def _fetch_friend_ids_paged(self, user_id, cursor=-1) -> (int, int, list):
        # http://python-twitter.readthedocs.io/en/latest/twitter.html#twitter.api.Api.GetFriendIDsPaged
        next_cursor, prev_cursor, friend_ids = self.api.GetFriendIDsPaged(
            user_id=user_id,
            cursor=cursor,
        )
        return next_cursor, prev_cursor, friend_ids


if __name__ == '__main__':
    API_TEST = True
    if API_TEST:
        user_network_api = UserNetworkAPIWrapper(
            config_file_path='./config.ini',
            event_path_list=get_event_files(),
            story_ids=[
                273182568298450945,
            ],
        )
        user_network_api.get_user_network()
    else:
        user_network = UserNetwork()
        user_network.load()
        print(user_network.user_id_to_friend_ids)
