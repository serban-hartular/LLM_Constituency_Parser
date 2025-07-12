
print('Importing...')
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, pipeline

import datasets

dataset_source = 'hartular/agreement-errors-model-results'

print('Downloading ds')
ds = datasets.load_dataset(dataset_source)
ds = ds['train']

ds_list= ds.to_list()
text_list = [ex['text'] for ex in ds_list]
text_list = text_list[:10]

levels = ['phrase', 'sentence']
features = ['case', 'gender', 'number', 'all']
task = 'text-classification'

for level in levels:
    for feature in features:
        model_name = f'label-{feature}_agreement-{level}-rrt'
        model_source = 'hartular/' + model_name
        print('Downloading next model')
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_source)
            tokenizer = AutoTokenizer.from_pretrained(model_source)
            mpipe = pipeline(task, model=model, tokenizer=tokenizer)
        except Exception as e:
            print(f'Error with model {model_name}: "{str(e)}"')
            continue
        print(f'Doing model {model_name}')
        model_results = mpipe(text_list, batch_size=4)
        model_results = [int(ex['label']=='good') for ex in model_results]
        for i, result in enumerate(model_results):
            ds_list[i][model_name] = result
        ds_up = datasets.Dataset.from_list(ds_list)
        print('Uploading results')
        try:
            ds_up.push_to_hub(dataset_source)
        except Exception as e:
            print(f'Error uploading data: {str(e)}')

