from collections import Counter

import pandas as pd
import datasets
import random

def to_train_test(datalist : list[dict], shuffle = False) -> (list[dict], list[dict]):
    """Split is approx 75/25. The assumption is that each data dict contains the items
    'good_sentence' and 'bad_sentence'"""
    good_sentences = Counter([d['good_sentence'] for d in datalist])
    good_sentences = list(good_sentences.items())
    good_sentences.sort(key=lambda t : -t[1]) # sort by count, decreasing
    good_sentences = [t[0] for t in good_sentences] # strings only
    good_sentence_test = good_sentences[::4]
    train_data, test_data = [], []
    for d in datalist:
        target_list = test_data if d['good_sentence'] in good_sentence_test else train_data
        target_list.append({'text':d['good_sentence'], 'label':1})
        target_list.append({'text': d['bad_sentence'], 'label': 0})
    if shuffle:
        random.shuffle(train_data)
        random.shuffle(test_data)
    return train_data, test_data

