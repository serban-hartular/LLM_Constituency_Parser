from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from datasets import Dataset, DatasetDict
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer


model_source = "dumitrescustefan/bert-base-romanian-cased-v1"
task = 'token-classification'

def tokenize_item(example : dict, tokenizer : AutoTokenizer, label2num : dict[str, int]):
    tokenized_input = tokenizer(example["words"], is_split_into_words=True)
    word_ids = tokenized_input.word_ids()
    labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None: # special token
            labels.append(-100)
            continue
        current_label = example['labels'][word_idx]
        if previous_word_idx == word_idx:
            current_label = 'I' + current_label[1:] # from B-HEAD to I-HEAD (or from I-HEAD to I-HEAD)
        labels.append(label2num[current_label])
        previous_word_idx = word_idx
    tokenized_input['labels'] = labels
    return tokenized_input

tokenizer = AutoTokenizer.from_pretrained(model_source)
dsd = DatasetDict.load_from_disk('./datasets/labelled_spans_head_deps-dev')

labels = ["0", 'B-DEP', 'I-DEP', 'B-HEAD', 'I-HEAD']
label2num = {l:i for i,l in enumerate(labels)}
num2label = {v:k for k,v in label2num.items()}

dsd_tokenized = dsd.map(lambda e : tokenize_item(e, tokenizer, label2num))


data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

model = AutoModelForTokenClassification.from_pretrained(
    model_source, num_labels=len(labels), id2label=num2label, label2id=label2num
)

training_args = TrainingArguments(
    output_dir="my_awesome_wnut_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    #eval_strategy="epoch",
    #save_strategy="epoch",
    #load_best_model_at_end=True,
    #push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dsd_tokenized["train"],
    #eval_dataset=tokenized_wnut["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    #compute_metrics=compute_metrics,
)

trainer.train()