import datasets

from generate_datasets import to_train_test

print('Importing')

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline

def load_pipeline(task : str, model_source : str) -> pipeline:
    model = AutoModelForSequenceClassification.from_pretrained(model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    return pipeline(task, model=model, tokenizer=tokenizer)


max_length = 512
NUM_EPOCHS = 3

labelled_text_ds_source = 'hartular/texts-labelled-grammaticality'
task = 'text-classification'
count = 5
grammatical_errors_source = 'hartular/rrt-grammatical_errors-v3'

print('Loading dataset')

original_data = datasets.load_dataset(grammatical_errors_source)
original_data = original_data['train']

filters = {
    'AgreeGender':
        "ex['error_family'] == 'morphology' and 'feature=Gender' in ex['misc']",
    'AgreeNumber':
        "ex['error_family'] == 'morphology' and 'feature=Number' in ex['misc']",
    'AgreePerson':
        "ex['error_family'] == 'morphology' and 'feature=Person' in ex['misc']",
}

filters.update({
    'AgreeGenNum' : f"({filters['AgreeGender']}) or ({filters['AgreeNumber']})",
    'AgreeGenPers' : f"({filters['AgreeGender']}) or ({filters['AgreePerson']})",
    'AgreeNumPers' : f"({filters['AgreeNumber']}) or ({filters['AgreePerson']})",
    'AgreeGenNumPers' : ' or '.join([f"({filters[x]})" for x in filters.keys()]),
})


def filter_fn(ex):
    pass


for filter_name, condition in filters.items():
    exec(f'def filter_fn(ex):\n\treturn {condition}\n')
    ds_filtered = original_data.filter(filter_fn)
    datalist = ds_filtered.to_list()
    train_list, test_list = to_train_test(datalist)
    ds_dict = datasets.DatasetDict()
    ds_dict['train'] = datasets.Dataset.from_list(train_list)
    ds_dict['test'] = datasets.Dataset.from_list(test_list)

    if count is not None:
        ds_dict['train'] = ds_dict['train'].select(range(count))

    model_source = "dumitrescustefan/bert-base-romanian-cased-v1"
    # dataset_source = 'hartular/agreement-errors-ro-rrt'
    # feature = 'Gender'
    model_name = f'hartular/rrt-ungramaticality-{filter_name}'
    destination_dir = f'./models/{model_name}'

    print(f'Task: {task}')
    print(f'Model source: {model_source}\nData filter: {filter_name}\nDestination dir: {destination_dir}\nNum epochs:{NUM_EPOCHS}')

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
        push_to_hub=True if count is not None else False
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
    mpipe = load_pipeline(task, model_name)


    labelled_texts_dsdict = datasets.load_dataset(labelled_text_ds_source)
    used_in_training = set(ds_dict['train']['text'])
    texts_to_label = set(original_data['good_text'])|set(original_data['bad_text'])|set(labelled_texts_dsdict['actual']['text'])
    texts_to_label = list(texts_to_label)
    texts_to_label.sort()
    is_train = ['train' if s in used_in_training else '' for s in texts_to_label]
    results = mpipe(texts_to_label, batch_size=4)
    results = [d['score'] if d['label'] == 'good' else 1-d['score'] for d in results]
    model_results = [{'text':t, 'score':s, 'use':u} for t,s,u in zip(texts_to_label, results, is_train)]
    labelled_texts_dsdict[model_name] = datasets.Dataset.from_list(model_results)
    labelled_texts_dsdict.push_to_hub(labelled_text_ds_source)

    # now check if 'actual' needs to be updated
    actual_texts = set(labelled_texts_dsdict['actual']['text'])
    not_done_texts = actual_texts.difference(texts_to_label)
    if not_done_texts:
        print('Adding texts to actual')
        good = set(original_data['good_text'])
        new_data = [{'text':t, 'use':'', 'score':int(t in good)} for t in not_done_texts]
        labelled_texts_dsdict['actual'] = datasets.concatenate_datasets(
            [labelled_texts_dsdict['actual'], datasets.Dataset.from_list(new_data)]
        )
        labelled_texts_dsdict.push_to_hub(labelled_text_ds_source)

    # now check if other models have undone texts
    for other_model_name in [s for s in labelled_texts_dsdict.keys() if s not in ('actual', model_name)]:
        other_model_texts = set(labelled_texts_dsdict[other_model_name]['text'])
        not_done_texts = other_model_texts.difference(texts_to_label)
        if not_done_texts:
            print(f'Adding texts to {other_model_name}')
            mpipe = load_pipeline(task, other_model_name)
            not_done_texts = list(not_done_texts)
            not_done_texts.sort()
            results = mpipe(not_done_texts, batch_size=4)
            results = [d['score'] if d['label'] == 'good' else 1 - d['score'] for d in results]
            new_data = [{'text': t, 'score': s, 'use': ''} for t, s in zip(not_done_texts, results)]
            labelled_texts_dsdict[other_model_name] = datasets.concatenate_datasets(
                [labelled_texts_dsdict[other_model_name], datasets.Dataset.from_list(new_data)]
            )
            labelled_texts_dsdict.push_to_hub(labelled_text_ds_source)


