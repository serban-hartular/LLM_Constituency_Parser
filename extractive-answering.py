import datasets

task = 'question-answering'
model_source = "dumitrescustefan/mt5-base-romanian"
dataset_source = 'hartular/dsdict_qa_head_dependent-bare-question-dev'
new_model_name = 'test-mt5' #'head_dependent-qa-t5-bare-question-dev'
destination_dir = f'./models/{new_model_name}'
# model_type = 't5' # t5, bert
max_length = 384

train_samples = 100 # -1 for all
test_samples = 25 # -1 for all

print(f'Task: {task}')
print(f"""Model source: {model_source}
Dataset source: {dataset_source}
Destination dir: {destination_dir}
Train/Test samples: {train_samples}/{test_samples}""")

print('Importing')

# if model_type not in ('bert', 't5', 'mt5'):
#     print(f'Unknown model type {model_type}, defaulting to bert')
#     model_type = 'bert'
#
# if model_type == 'bert':
#     from transformers import AutoModelForQuestionAnswering, AutoTokenizer
#     AUTOMODEL = AutoModelForQuestionAnswering
#     AUTOTOKENIZER = AutoTokenizer
# elif model_type == 't5':
#     from transformers import T5ForQuestionAnswering, T5TokenizerFast
#     AUTOMODEL = T5ForQuestionAnswering
#     AUTOTOKENIZER = T5TokenizerFast
# elif model_type == 'mt5':
#     from transformers import MT5ForQuestionAnswering, MT5TokenizerFast
#     AUTOMODEL = MT5ForQuestionAnswering
#     AUTOTOKENIZER = MT5TokenizerFast

from transformers import AutoModelForQuestionAnswering, AutoTokenizer
AUTOMODEL = AutoModelForQuestionAnswering
AUTOTOKENIZER = AutoTokenizer

from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding
# import evaluate
# import numpy as np
from transformers import TrainingArguments, Trainer



print('Loading dataset')

# ds_dict = DatasetDict.load_from_disk(dataset_source)
ds_dict = datasets.load_dataset(dataset_source)
# labels = list(set(ds_dict['train']['label']) | set(ds_dict['test']['label']))
# labels.sort()
# id2label = {i:l for i,l in enumerate(labels)}
# label2id = {l:i for i,l in id2label.items()}
# ds_dict = ds_dict.map(lambda ex : {'text':ex['text'], 'label':label2id[ex['label']]})



print('Loading tokenizer')

tokenizer = AUTOTOKENIZER.from_pretrained(model_source)
def preprocess_function(examples, _tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = _tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# accuracy = evaluate.load("accuracy")
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return accuracy.compute(predictions=predictions, references=labels)

print('Tokenizing dataset')

if train_samples > 0:
    ds_dict['train'] = ds_dict['train'].select(range(train_samples))
if test_samples > 0:
    ds_dict['test'] = ds_dict['test'].select(range(test_samples))

tokenized_dsd = ds_dict.map(lambda ex : preprocess_function(ex, tokenizer), batched=True, remove_columns=ds_dict["train"].column_names)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


print('Loading model')

model = AUTOMODEL.from_pretrained(
    model_source
)

print('Configuring trainer')

training_args = TrainingArguments(
    output_dir=destination_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    # load_best_model_at_end=True,
    push_to_hub=False, # True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dsd["train"],
    eval_dataset=tokenized_dsd["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
)

print('Training')

trainer.train()

# from transformers import pipeline
# p = pipeline(task, model=model, tokenizer=tokenizer)

