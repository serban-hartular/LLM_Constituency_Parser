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


def load_pipeline(model_source : str) -> pipeline:
    model = AutoModelForSequenceClassification.from_pretrained(model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    return pipeline(task, model=model, tokenizer=tokenizer)

def ds_to_predict_actual(ds : Dataset, mpipe : pipeline, ds_to_model_labels : dict = None) -> (list, list):
    actual = ds['label']
    if ds_to_model_labels:
        actual = [ds_to_model_labels[a] for a in actual]
    predicted = mpipe(ds['text'])
    predicted = [p['label'] for p in predicted]
    return predicted, actual


if __name__ == "__main__":
    levels = ['phrase', 'sentence']
    features = ['case', 'gender', 'number', 'all']
    task = 'text-classification'

    model_names = [f'label-{feature}_agreement-{level}-rrt' for level in levels for feature in features]

    test_sets = {model_name: datasets.load_dataset('hartular/'+model_name)['test'].to_list() for model_name in model_names}
    results = datasets.load_dataset('hartular/agreement-errors-model-results')['train']
    results = {ex['text']:ex for ex in results.to_list()}

    model_stats = defaultdict(dict)

    for model in model_names:
        for ds_name, ds_test in test_sets.items():
            label_list = [(ex['label'], results[ex['text']][model]) for ex in ds_test if ex['text'] in results]
            actual, predicted = zip(*label_list) # unzip
            actual, predicted = list(actual), list(predicted)
            raw_score = sum([a==p for (a,p) in label_list]) / len(label_list)
            scores = get_scores(predicted, actual, 0) # looking for mistakes
            scores['raw'] = raw_score
            model_stats[model + ' model'][ds_name + ' data'] = scores


