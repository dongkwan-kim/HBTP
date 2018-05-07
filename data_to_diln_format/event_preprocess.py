import numpy as np
import pandas as pd
from collections import defaultdict

events = pd.read_csv('./../data/event/synchronized/event_table_twitter15_2018-04-30 15:01:39.183091.csv')

parent_child = defaultdict(list)
child_parent = defaultdict(list)
story_contributors = defaultdict(list)
user_stories = defaultdict(list)

for i in events.index:
    event = events.loc[i]
    parent, user, story = str(event['parent_id']), str(event['user_id']), str(event['story_id'])

    story_contributors[story].append(user)
    user_stories[user].append(story)
    # if parent != 'ROOT':
    parent_child[parent].append(user)
    child_parent[user].append( (parent, story) )

    # if parent != 'ROOT' and parent not in story_contributors[story]: print( story, user, parent )
    # else:  print('b')
    if i % 1000 == 0: print(i)

parent_child, child_parent, story_contributors, user_stories = dict(parent_child), dict(child_parent), dict(story_contributors), dict(user_stories)
    
leaf_users = set()

for k, v in parent_child.items():
    for leaf_user in v:
        if leaf_user not in parent_child and len(user_stories[leaf_user]) == 1:
            leaf_users.add(leaf_user)

parent_child_final = {k: [vv for vv in v if vv not in leaf_users] for k, v in parent_child.items()}
parent_child_final = {k:v for k, v in parent_child_final.items() if len(v) != 0}
child_parent_final = {k: v for k, v in child_parent.items() if k not in leaf_users} # we use this!
story_contributors_final = {k: [vv for vv in v if vv not in leaf_users] for k, v in story_contributors.items()} # we use this!
user_stories_final = {v : user_stories[v] for v in user_stories if v not in leaf_users}


