from format_story import *
from Main_TD_RvNN import *


if __name__ == '__main__':
    stories = get_formatted_stories(data_path='../data')
    eid_pool = list(stories.story_to_id.keys())
    run_wrapper(
        path_root='./Rumor_RvNN',
        _fold='2',
        _eid_pool=eid_pool,
    )
