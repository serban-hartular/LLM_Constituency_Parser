from collections import Counter
import random

import datasets

print('Importing')

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
import itertools

def load_pipeline(task : str, model_source : str) -> pipeline:
    model = AutoModelForSequenceClassification.from_pretrained(model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    return pipeline(task, model=model, tokenizer=tokenizer)

def set_to_mask(int_set : set[int]) -> int:
    mask = 0x0001
    value = 0x0000
    max_value = max(int_set)
    for bit in range(0, max_value+1):
        if bit in int_set:
            value = value|mask
        mask = mask << 1
    return value


def to_train_test(datalist : list[dict], shuffle = True) -> (list[dict], list[dict], set[str]):
    """Split is approx 75/25. The assumption is that each data dict contains the items
    'original' 'text', and 'error_class' """
    original_texts = Counter([d['original'] for d in datalist])
    original_texts = list(original_texts.items())
    original_texts.sort(key=lambda t : -t[1]) # sort by count, decreasing
    original_texts = [t[0] for t in original_texts] # strings only
    originals_test = original_texts[::4]
    originals_used_in_training = [t for t in original_texts if t not in originals_test]
    train_data, test_data = [], []
    for d in datalist:
        target_list = train_data if d['original'] in originals_used_in_training else test_data
        # to keep data balanced, repeat datum by num of classes it applies to
        target_list.extend([{'text':d['text'], 'label':d['actual']}] * d['error_mask'].bit_count())
    if shuffle:
        random.shuffle(train_data)
        random.shuffle(test_data)
    return train_data, test_data, originals_used_in_training


max_length = 512
NUM_EPOCHS = 2

results_ds_source = 'hartular/rrt-grammaticality-results-v0'
task = 'text-classification'
count = None

print('Loading dataset')

original_data = datasets.load_dataset(results_ds_source)
original_data = original_data['train']

morpho_agreement_mask   = 0b00000000000111
morpho_requirement_mask = 0b00000000111000
position_mask           = 0b00001111000000
omission_mask           = 0b11110000000000

_M = [morpho_agreement_mask, morpho_requirement_mask, position_mask, omission_mask]
mask_combos = [_M[0], _M[1], _M[2], _M[3],
          _M[0]|_M[1], _M[0]|_M[2], _M[0]|_M[3], _M[1]|_M[2], _M[1]|_M[3], _M[2]|_M[3],
          _M[0]|_M[1]|_M[2], _M[0]|_M[1]|_M[3], _M[0]|_M[2]|_M[3], _M[1]|_M[2]|_M[3],
          _M[0]|_M[1]|_M[2]|_M[3]]

for mask in mask_combos:
    ds_filtered = original_data.filter(lambda ex: ex['error_mask'] & mask)
    datalist = ds_filtered.to_list()
    # update mask in data (for counting bits in to_train_test()
    for d in datalist:
        d['error_mask'] = d['error_mask'] & mask
    train_list, test_list, originals_used_in_training = to_train_test(datalist)
    ds_dict = datasets.DatasetDict()
    ds_dict['train'] = datasets.Dataset.from_list(train_list)
    ds_dict['test'] = datasets.Dataset.from_list(test_list)

    if count is not None:
        ds_dict['train'] = ds_dict['train'].select(range(count))
    model_source = "dumitrescustefan/bert-base-romanian-cased-v1"
    # dataset_source = 'hartular/agreement-errors-ro-rrt'
    # feature = 'Gender'
    model_name = f'rrtUngramaticality{mask:04X}'
    destination_dir = f'./models/{model_name}'

    print(f'Task: {task}')
    print(f'Model source: {model_source}\nData filter: {mask:04X}\nDestination dir: {destination_dir}\nNum epochs:{NUM_EPOCHS}')

    labels = ['bad', 'good']
    id2label = {i:l for i,l in enumerate(labels)}
    label2id = {l:i for i,l in id2label.items()}

    # ds = ds.map(lambda ex : {'text':ex['text'], 'label':label2id[ex['label']]})

    print('Loading tokenizer')

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    print('Tokenizing dataset')

    tokenized_dsd = ds_dict.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    print('Loading model')

    model = AutoModelForSequenceClassification.from_pretrained(
        model_source, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    print('Configuring trainer')

    training_args = TrainingArguments(
        output_dir=destination_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dsd["train"],
        eval_dataset=tokenized_dsd["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
 )

    print('Training')

    trainer.train()

    # now, we get model outputs for all texts
    mpipe = load_pipeline(task, 'hartular/'+model_name)
    ds_to_label = original_data.filter(lambda ex: ex['original'] not in originals_used_in_training)
    texts_to_label = list(set(ds_to_label['text']))
    print('Labelling.')
    if count is not None:
        texts_to_label = texts_to_label[:count]
    results = mpipe(texts_to_label, batch_size=8)
    print('Done labelling.')
    results = {txt:int(r['label']=='good') for txt, r in zip(texts_to_label, results)}
    original_data = original_data.map(lambda ex: {model_name:results[ex['text']] if ex['text'] in results else -1})
    original_data.push_to_hub(results_ds_source)
