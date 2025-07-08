from typing import Any

import datasets

from datasets import Dataset

task = 'text-classification'
print('Importing')

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, pipeline

def get_scores(predicted : list, actual : list, good_label : Any) -> dict:
    tuple_list = list(zip(predicted, actual))
    num_positive = sum([a==good_label for a in actual if a == good_label])
    true_positive = sum([p==good_label for p,a in tuple_list if a == good_label])
    false_positive = sum([p==good_label for p,a in tuple_list if a != good_label])

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
    model_source = "hartular/label-all_agreement-phrase-rrt"
    print('Downloading model')
    mpipe = load_pipeline(model_source)
    print('Downloading dataset')
    ds_dict = datasets.load_dataset(model_source)
    print('Getting predicted values')
    pred, actual = ds_to_predict_actual(ds_dict['test'], mpipe, {0:'bad', 1:'good'})
    print('Scores for finding bad / good:')
    print(get_scores(pred, actual, 'bad'))
    print(get_scores(pred, actual, 'good'))
