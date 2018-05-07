import numpy as np
from nltk import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import re
from collections import Counter

stories = pd.read_csv('./../data/story/preprocessed/story_table_twitter15_2018-04-23 23:03:53.090923.csv')

stopwords = 'i, me, my, myself, we, our, ours, ourselves, you, your, yours, yourself, yourselves, he, him, his, himself, she, her, hers, herself, it, its, itself, they, them, their, theirs, themselves, what, which, who, whom, this, that, these, those, am, is, are, was, were, be, been, being, have, has, had, having, do, does, did, doing, a, an, the, and, but, if, or, because, as, until, while, of, at, by, for, with, about, against, between, into, through, during, before, after, above, below, to, from, up, down, in, out, on, off, over, under, again, further, then, once, here, there, when, where, why, how, all, any, both, each, few, more, most, other, some, such, no, nor, not, only, own, same, so, than, too, very, s, t, can, will, just, don, should, now'

id_content = dict()
delimiters = '\s'
ps = PorterStemmer()
wf = dict()

for i in range(len(stories)):
    row = stories.ix[i]
    content = row['title'] + '\n' + row['content']
    content = content.lower()
    words = re.split(delimiters, content)
    words = [ps.stem(v) for v in words]
    words = [v for v in words if v not in stopwords]
    words = [re.sub('[\W_]+', '', v) for v in words]
    words = [v for v in words if len(v) > 1]
    for word in words:
        if word not in wf:
            wf[word] = 0
        wf[word] += 1
    id_content[i] = words
    if i % 10 == 0: print (i)

freq_word = dict()
for word in wf.keys():
    freq = wf[word]
    if freq not in freq_word:
        freq_word[freq] = list()
    freq_word[freq].append(word)

for i in range(len(stories)):
    words_prev = id_content[i]
    words = [word for word in words_prev if 2 < wf[word] < 500]
    id_content[i] = words

vocab = set()
for words in id_content.values():
    vocab = vocab | set(words)

word2id = {v:k for k, v in enumerate(sorted(vocab))}
word_ids_dict = {i: [word2id[word] for word in id_content[i]  ] for i in range(len(id_content))}

word_ids = [np.array(list(Counter(word_ids_dict[i]).keys())) for i in range(len(word_ids_dict))]
word_cnt = [np.array(list(Counter(word_ids_dict[i]).values())) for i in range(len(word_ids_dict))]
id2word = {v:k for k, v in word2id.items()}
