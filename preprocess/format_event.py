import pandas as pd
from collections import defaultdict
import os
import pprint


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

    def pprint(self):
        pprint.pprint(self.__dict__)

    def get_formatted(self):
        events = pd.read_csv(self.event_path)

        parent_to_child = defaultdict(list)
        child_to_parent_and_story = defaultdict(list)
        story_to_users = defaultdict(list)
        user_to_stories = defaultdict(list)

        # Construct a dict from feature to feature
        for i in events.index:
            event = events.loc[i]
            parent, user, story = map(str, [event['parent_id'], event['user_id'], event['story_id']])

            parent_to_child[parent].append(user)
            child_to_parent_and_story[user].append((parent, story))
            story_to_users[story].append(user)
            user_to_stories[user].append(story)

            if i % 1000 == 0 and __name__ == '__main__':
                print(i)

        parent_to_child, child_to_parent_and_story, story_to_users, user_to_stories \
            = map(dict, [parent_to_child, child_to_parent_and_story, story_to_users, user_to_stories])

        # Construct a set of leaf users
        leaf_users = set()
        for parent, child in parent_to_child.items():
            for leaf_user in child:
                if leaf_user not in parent_to_child and len(user_to_stories[leaf_user]) == 1:
                    leaf_users.add(leaf_user)

        # Remove leaf users
        parent_to_child_final = {k: [vv for vv in v if vv not in leaf_users] for k, v in parent_to_child.items()}
        self.parent_to_child = {k: v for k, v in parent_to_child_final.items() if len(v) != 0}
        self.user_to_stories = {k: v for k, v in user_to_stories.items() if k not in leaf_users}
        self.child_to_parent_and_story = {k: v for k, v in child_to_parent_and_story.items() if k not in leaf_users}
        self.story_to_users = {k: [vv for vv in v if vv not in leaf_users] for k, v in story_to_users.items()}


def get_formatted_events():
    r_list = []
    for event_path in get_event_files():
        fe = FormattedEvent(event_path)
        fe.get_formatted()
        r_list.append(fe)
    return r_list


if __name__ == '__main__':
    for data in get_formatted_events():
        data.pprint()
