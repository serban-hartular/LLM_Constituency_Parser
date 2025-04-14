task = 'text-classification'
model_source = "dumitrescustefan/bert-base-romanian-cased-v1"
dataset_source = './datasets/dsdict_context-dev'
destination_dir = './models/bert-base-context-dev'

print(f'Task: {task}')
print(f'Model source: {model_source}\nDataset source: {dataset_source}\nDestination dir: {destination_dir}')

print('Importing')

from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer



print('Loading dataset')

ds_dict = DatasetDict.load_from_disk(dataset_source)
labels = list(set(ds_dict['train']['label']) | set(ds_dict['test']['label']))
labels.sort()
id2label = {i:l for i,l in enumerate(labels)}
label2id = {l:i for i,l in id2label.items()}
ds_dict = ds_dict.map(lambda ex : {'text':ex['text'], 'label':label2id[ex['label']]})



print('Loading tokenizer')

tokenizer = AutoTokenizer.from_pretrained(model_source)
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

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
    push_to_hub=False, # True
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

