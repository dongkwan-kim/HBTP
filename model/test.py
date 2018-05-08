import use_preprocess as up
from format_event import *
from format_story import *

import diln
import hdp
import hbtp_upstream

up.use_preprocess()


def get_formatted():
    formatted_events = get_formatted_events()
    formatted_stories = get_formatted_stories()
    return formatted_events, formatted_stories


def get_corpus(f_events: FormattedEvent, f_stories: FormattedStory):
    return hbtp_upstream.Corpus(
        vocab=[f_stories.id_to_word[i] for i in range(len(f_stories.id_to_word))],
        word_ids=f_stories.word_ids,
        word_cnt=f_stories.word_cnt,
        child_to_parent_and_story=f_events.child_to_parent_and_story,
        story_to_users=f_events.story_to_users,
        n_topic=100,
    )


def run_hbtp_upsream(corpus: hbtp_upstream.Corpus):
    hbtp = hbtp_upstream.HBTP(
        n_topic=100,
        n_voca=len(corpus.vocab),
    )
    hbtp.fit(corpus)
    return hbtp


if __name__ == '__main__':
    events, stories = get_formatted()
    for e, s in zip(events, stories):
        run_hbtp_upsream(get_corpus(e, s))
        break
