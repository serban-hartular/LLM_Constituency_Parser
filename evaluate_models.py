import datasets

model_result_source = 'hartular/texts-labelled-grammaticality'
data_source = 'hartular/rrt-grammatical_errors-v3'

result_dsd = datasets.load_dataset(model_result_source)

data_ds = datasets.load_dataset(data_source)['train']

filters = {
    'AgreeGender':
        "ex['error_family'] == 'morphology' and 'feature=Gender' in ex['misc']",
    'AgreeNumber':
        "ex['error_family'] == 'morphology' and 'feature=Number' in ex['misc']",
    'AgreePerson':
        "ex['error_family'] == 'morphology' and 'feature=Person' in ex['misc']",
}

def filter_fn(ex):
    pass

# actual_results = {d['text']:d['score'] for d in result_dsd['actual'].to_list()}

model_scores = {}

for model_name in [k for k in result_dsd.keys() if k != 'actual']:
    model_scores[model_name] = {}
    model_results_ds = result_dsd[model_name]
    train_texts = set(model_results_ds.filter(lambda d: d['use']=='train')['text'])
    model_results = {d['text']:d['score'] for d in model_results_ds.to_list()}
    for data_range_name, condition in filters.items():
        # select data to try it on
        exec(f'def filter_fn(ex):\n\treturn {condition}\n')
        data_range = data_ds.filter(filter_fn)
        # filter to avoid texts used in training
        data_range = data_range.filter(lambda ex: ex['good_text'] not in train_texts and ex['bad_text'] not in train_texts)
        # extract good texts and bad texts
        text_lists = {'good_text':[], 'bad_text':[]}
        for label in text_lists:
            text_lists[label] = data_range[label]
        # get model results
        predicted_results = {'good_text':[], 'bad_text':[]}
        for label in predicted_results:
            predicted_results[label] = [0 if model_results[t] < 0.5 else 1 for t in text_lists[label]]
        scores = {}
        for label in predicted_results:
            scores[label] = sum(predicted_results[label]) if label == 'good_text' else (
                len(predicted_results[label]) - sum(predicted_results[label])
            )
            scores[label] = scores[label] / len(predicted_results[label])
        scores['sample_size'] = len(data_range)
        model_scores[model_name][data_range_name] = scores

