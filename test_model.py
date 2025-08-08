from collections import defaultdict
from typing import Any

import datasets

from datasets import Dataset, DatasetDict

task = 'text-classification'
print('Importing')


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, pipeline

def get_scores(predicted : list, actual : list, good_label : Any) -> dict:
    tuple_list = list(zip(predicted, actual))
    num_positive = sum([a==good_label for a in actual if a == good_label])
    true_positive = sum([p==good_label for p,a in tuple_list if a == good_label])
    false_positive = sum([p==good_label for p,a in tuple_list if a != good_label])

    if num_positive == 0 or true_positive+false_positive == 0:
        return {'precision':0.0, 'recall':0.0, 'F':0.0}

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_positive

    return {'precision':precision, 'recall':recall, 'F':2/(1/precision + 1/recall)}


def load_pipeline(task : str, model_source : str) -> pipeline:
    model = AutoModelForSequenceClassification.from_pretrained(model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    return pipeline(task, model=model, tokenizer=tokenizer)

def ds_to_predict_actual(ds : Dataset, mpipe : pipeline, ds_to_model_labels : dict = None) -> (list, list):
    actual = ds['label']
    if ds_to_model_labels:
        actual = [ds_to_model_labels[a] for a in actual]
    predicted = mpipe(ds['text'], batch_size=4)
    predicted = [p['label'] for p in predicted]
    return predicted, actual


if __name__ == "__main__":
    task = 'text-classification'
    feature = 'gender'
    model_name = f'hartular/label-{feature}_agreement-sentence-rrt-v2'

    print('Doing ' + model_name)

    print('Loading datasets')
    ds_dict = datasets.load_dataset(model_name)

    print('Loading model')
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    mpipe = load_pipeline(task, model_name)

