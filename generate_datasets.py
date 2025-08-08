
import pickle
import datasets
from collections import Counter, defaultdict
import random

def to_train_test(datalist : list[dict], shuffle = False) -> (list[dict], list[dict]):
    """Split is approx 75/25. The assumption is that each data dict contains the items
    'good_sentence' and 'bad_sentence'"""
    good_sentences = Counter([d['good_text'] for d in datalist])
    good_sentences = list(good_sentences.items())
    good_sentences.sort(key=lambda t : -t[1]) # sort by count, decreasing
    good_sentences = [t[0] for t in good_sentences] # strings only
    good_sentence_test = good_sentences[::4]
    train_data, test_data = [], []
    for d in datalist:
        target_list = test_data if d['good_text'] in good_sentence_test else train_data
        target_list.append({'text':d['good_text'], 'label':1})
        target_list.append({'text': d['bad_text'], 'label': 0})
    if shuffle:
        random.shuffle(train_data)
        random.shuffle(test_data)
    return train_data, test_data



if __name__ == "__main__":
    original_data = datasets.load_dataset('hartular/rrt-grammatical_errors-v3')
    original_data = original_data['train']

    filters = {
        'AgreeGender': "\treturn ex['error_family'] == 'morphology' and 'feature=Gender' in ex['misc']",
        'AgreeNumber': "\treturn ex['error_family'] == 'morphology' and 'feature=Number' in ex['misc']",
        'AgreePerson': "\treturn ex['error_family'] == 'morphology' and 'feature=Person' in ex['misc']",
    }

    def filter_fn(ex):
        pass

    for name, condition in filters.items():
        exec(f'def filter_fn(ex):\n{condition}\n')
        ds = original_data.filter(filter_fn)
        datalist = ds.to_list()
        train_list, test_list = to_train_test(datalist)
        ds_dict = datasets.DatasetDict()
        ds_dict['train'] = datasets.Dataset.from_list(train_list)
        ds_dict['test'] = datasets.Dataset.from_list(test_list)


