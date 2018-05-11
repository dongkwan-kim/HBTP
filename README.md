# Homogeneity-Based Transmissive Process (HBTP)

## Directory Structure
```
HBTP
├── README.md
├── data
│   ├── README.md
│   ├── event
│   │   ├── raw
│   │   └── synchronized
│   ├── stopsentences.txt
│   ├── stopwords.txt
│   └── story
│       ├── explicit-error-preprocessed
│       ├── implicit-error-preprocessed
│       ├── preprocessed
│       └── raw
├── model
│   ├── RBFKernel.py
│   ├── corpus.py
│   ├── diln.py
│   ├── hbtp.py
│   ├── hbtp_transmissive.py
│   ├── hbtp_upstream.py
│   ├── hdp.py
│   ├── model.py
│   ├── test.py
│   └── use_preprocess.py
├── preprocess
│   ├── WriterWrapper.py
│   ├── config.ini
│   ├── crawler.py
│   ├── format_event.py
│   ├── format_story.py
│   ├── parser.py
│   ├── preprocess.py
│   ├── split_train_test.py
│   ├── stats.py
│   └── synch.py
├── requirements.txt
└── rumor_detection_acl2017
    ├── README.txt
    ├── twitter15
    │   ├── label.txt
    │   └── tree
    ├── twitter16
    │   ├── label.txt
    │   └── tree
    └── twittertest
        └── label.txt
```

## References
- Liu, Xiaomo, et al. "Real-time rumor debunking on twitter." Proceedings of the 24th ACM International on Conference on Information and Knowledge Management. ACM, 2015.
- Ma, Jing, et al. "Detecting Rumors from Microblogs with Recurrent Neural Networks." IJCAI. 2016.
- Ma, Jing, Wei Gao, and Kam-Fai Wong. "Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning." Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Vol. 1. 2017.
