import pandas as pd
from collections import defaultdict
import os
import pprint
import pickle

EVENT_PATH = '../data/event/synchronized'


def get_event_files():
    return [os.path.join(EVENT_PATH, f) for f in os.listdir(EVENT_PATH) if 'csv' in f]


class FormattedEvent:

    def __init__(self, event_path):
        self.event_path = event_path
        self.parent_to_child = None
        self.child_to_parent_and_story = None
        self.story_to_users = None
        self.user_to_stories = None

    def get_twitter_year(self):
        return self.event_path.split('_')[2]

    def pprint(self):
        pprint.pprint(self.__dict__)

    def indexify(self, target_dict: dict, key_to_id: dict, value_to_id: dict, is_c2ps=False):
        """
        :param target_dict: dict {key -> list of values}
        :param key_to_id: dict
        :param value_to_id: dict
        :param is_c2ps: is_child_to_parent_and_story
        :return: dict {key_to_id[key] -> value_to_id[value]}
        """
        r_dict = {}
        for key, values in target_dict.items():
            if not is_c2ps:
                r_dict[key_to_id[key]] = list(map(lambda v: value_to_id[v], values))
            else:
                # c2ps: key:user -> (key:user, value:story)
                r_dict[key_to_id[key]] = list(map(lambda v: (key_to_id[v[0]], value_to_id[v[1]]), values))
        return r_dict

    def dump(self):
        file_name = 'FormattedEvent_{}.pkl'.format(self.get_twitter_year())
        with open(os.path.join(EVENT_PATH, file_name), 'wb') as f:
            pickle.dump(self, f)
        print('Dumped: {0}'.format(file_name))

    def load(self):
        file_name = 'FormattedEvent_{}.pkl'.format(self.get_twitter_year())
        try:
            with open(os.path.join(EVENT_PATH, file_name), 'rb') as f:
                loaded = pickle.load(f)
                self.parent_to_child = loaded.parent_to_child
                self.child_to_parent_and_story = loaded.child_to_parent_and_story
                self.story_to_users = loaded.story_to_users
                self.user_to_stories = loaded.user_to_stories
            print('Loaded: {0}'.format(file_name))
            return True
        except:
            print('Load Failed: {0}'.format(file_name))
            return False

    def get_formatted(self):

        if self.load():
            return

        events = pd.read_csv(self.event_path)

        parent_to_child = defaultdict(list)
        child_to_parent_and_story = defaultdict(list)
        story_to_users = defaultdict(list)
        user_to_stories = defaultdict(list)

        user_set = set()
        story_set = set()

        # Construct a dict from feature to feature
        for i in events.index:
            event = events.loc[i]
            parent, user, story = map(str, [event['parent_id'], event['user_id'], event['story_id']])

            parent_to_child[parent].append(user)
            child_to_parent_and_story[user].append((parent, story))
            story_to_users[story].append(user)
            user_to_stories[user].append(story)

            user_set.update([parent, user])
            story_set.add(story)

            if i % 1000 == 0 and __name__ == '__main__':
                print(i)

        # Construct a set of leaf users
        leaf_users = set()
        for parent, child in parent_to_child.items():
            for leaf_user in child:
                if leaf_user not in parent_to_child and len(user_to_stories[leaf_user]) == 1:
                    leaf_users.add(leaf_user)

        # Remove leaf users
        parent_to_child_final = {k: [vv for vv in v if vv not in leaf_users] for k, v in parent_to_child.items()}
        parent_to_child = {k: v for k, v in parent_to_child_final.items() if len(v) != 0}
        user_to_stories = {k: v for k, v in user_to_stories.items() if k not in leaf_users}
        child_to_parent_and_story = {k: v for k, v in child_to_parent_and_story.items() if k not in leaf_users}
        story_to_users = {k: [vv for vv in v if vv not in leaf_users] for k, v in story_to_users.items()}
        user_set = set(u for u in user_set if u not in leaf_users)

        user_to_id = dict((user, idx) for idx, user in enumerate(sorted(user_set)))
        story_to_id = dict((story, idx) for idx, story in enumerate(sorted(story_set)))

        # Indexify
        self.parent_to_child = self.indexify(parent_to_child, user_to_id, user_to_id)
        self.child_to_parent_and_story = self.indexify(child_to_parent_and_story, user_to_id, story_to_id, is_c2ps=True)
        self.story_to_users = self.indexify(story_to_users, story_to_id, user_to_id)
        self.user_to_stories = self.indexify(user_to_stories, user_to_id, story_to_id)


def get_formatted_events():
    r_list = []
    for event_path in get_event_files():
        fe = FormattedEvent(event_path)
        fe.get_formatted()
        r_list.append(fe)
    return r_list


if __name__ == '__main__':
    for data in get_formatted_events():
        data.dump()
