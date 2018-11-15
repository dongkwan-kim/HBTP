# Homogeneity-Based Transmissive Process (HBTP)

This is the code and dataset repository of "Homogeneity-Based Transmissive Process To Model True and False News in Social Networks (WSDM 2019)".

## Directory Structure
```
HBTP
├── README.md
├── data
│   ├── README.md
│   ├── event
│   │   ├── raw
│   │   └── synchronized
│   │       └── FormattedEvent_twitter1516.pkl
│   ├── stopsentences.txt
│   ├── stopwords.txt
│   └── story
│       ├── explicit-error-preprocessed
│       ├── implicit-error-preprocessed
│       ├── preprocessed
│       ├── preprocessed-label
│       │   └── FormattedStory_twitter1516.pkl
│       └── raw
├── model
│   ├── hbtp.py
│   ├── RBFKernel.py
│   ├── corpus.py
│   ├── model.py
│   ├── test.py
│   └── use_preprocess.py
├── preprocess
│   ├── TwitterAPIWrapper.py
│   ├── WriterWrapper.py
│   ├── config.ini
│   ├── crawler.py
│   ├── format_event.py
│   ├── format_story.py
│   ├── label.py
│   ├── network.py
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

## Run
1. Install `requirements.txt`.
2. Run `model/test.py`. To change the model class, replace the first argument of `run_model()`

## Bibtex
```
@inproceedings{kim2019homogeneity,
  title={Homogeneity-Based Transmissive Process To Model True and False News in Social Networks},
  author={Kim, Jooyeon and Kim, Dongkwan and Oh, Alice},
  booktitle={Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining},
  pages={/* TODO */},
  year={2019},
  organization={ACM}
}
```

## Code References
- Code base: https://github.com/dongwookim-ml/python-topic-model
- `model/RBFKernel.py`: https://github.com/melihkandemir/gpstm

## Dataset References
- If you use dataset in `data/story` or `data/event`, please cite our work (Kim et al.) and precendence works below.
- You can directly get `rumor_detection_acl2017` from [here](https://github.com/majingCUHK/Rumor_RvNN)

```
@inproceedings{liu2015real,
  title={Real-time Rumor Debunking on Twitter},
  author={Liu, Xiaomo and Nourbakhsh, Armineh and Li, Quanzhi and Fang, Rui and Shah, Sameena},
  booktitle={Proceedings of the 24th ACM International on Conference on Information and Knowledge Management},
  pages={1867--1870},
  year={2015}
}
@inproceedings{ma2016detecting,
  title={Detecting Rumors from Microblogs with Recurrent Neural Networks},
  author={Ma, Jing and Gao, Wei and Mitra, Prasenjit and Kwon, Sejeong and Jansen, Bernard J. and Wong, Kam-Fai and Meeyoung, Cha},
  booktitle={The 25th International Joint Conference on Artificial Intelligence},
  year={2016},
  organization={AAAI}
}
@inproceedings{ma2017detect,
  title={Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning},
  author={Ma, Jing and Gao, Wei and Wong, Kam-Fai},
  booktitle={The 55th annual meeting of the Association for Computational Linguistics},
  pages={xxx-xxx},
  year={2017},
  organization={Association for Computational Linguistics}
}
```