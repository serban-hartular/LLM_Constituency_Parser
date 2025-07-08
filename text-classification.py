import datasets
print('Importing')

from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

max_length = 512
level = 'sentence' # or phrase

for feature in ['Gender', 'Number', 'Case', 'All']:

    task = 'text-classification'
    model_source = "dumitrescustefan/bert-base-romanian-cased-v1"
    dataset_source = 'hartular/agreement-errors-ro-rrt'
    # feature = 'Gender'
    model_name = f'label-{feature.lower()}_agreement-{level}-rrt'
    destination_dir = f'./models/{model_name}'

    print(f'Task: {task}')
    print(f'Model source: {model_source}\nDataset source: {dataset_source}\nDestination dir: {destination_dir}')





    print('Loading dataset')

    count = None

    ds_orig = datasets.load_dataset(dataset_source)
    ds_orig = ds_orig['train']
    ds_use = ds_orig.filter(lambda d : d['feature']==feature if feature and feature.lower() != 'all' else True)
    ds_use = ds_use.shuffle()

    # separate into good and bad. don't reuse sentences
    text_set = set()
    labelled_data = []
    ds_use = ds_use.to_list()
    for d in ds_use:
        if d[f'good_{level}'] in text_set:
            continue
        text_set.add(d[f'good_{level}'])
        for key, label in zip([f'good_{level}', f'good_{level}'], [1, 0]):
            labelled_data.append({'text':d[key], 'label':label})

    ds = Dataset.from_list(labelled_data)
    ds = ds.shuffle()
    if count:
        ds = ds.select(range(count))

    ds_dict = ds.train_test_split(test_size=0.25)
    print(f'Training dataset len: {len(ds_dict["train"])}')


    labels = ['bad', 'good']
    id2label = {i:l for i,l in enumerate(labels)}
    label2id = {l:i for i,l in id2label.items()}

    # ds_dict = ds_dict.map(lambda ex : {'text':ex['text'], 'label':label2id[ex['label']]})



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
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True # False, # True
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

    from transformers import pipeline

    p = pipeline(task, model=trainer.model, tokenizer=trainer.processing_class)

    ds_dict.push_to_hub(model_name)
