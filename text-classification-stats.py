import datasets

model_name_0 = 'rrtUngramaticality0007'
results_source = 'hartular/rrt-grammaticality-results-v0'
num_classes = 13
error_masks = [0x1 << b for b in range(0, num_classes+1)]


ds_all_results = datasets.load_dataset(results_source)['train']

model_stats = {}

for model_name in ds_all_results.column_names[5:]:
    model_stats[model_name] = {}
    ds_model_results = ds_all_results.filter(lambda ex: ex[model_name] != -1)
    for error_class in range(0, num_classes+1):
        class_results = [d for d in ds_model_results.to_list() if d['error_mask'] & error_masks[error_class]]
        good = [(d['actual'], d[model_name]) for d in class_results if d['actual']]
        bad  = [(d['actual'], d[model_name]) for d in class_results if not d['actual']]
        good_score = sum([a==p for a,p in good]) / len(good)
        bad_score = sum([a==p for a,p in bad]) / len(bad)
        model_stats[model_name][error_class] = {'recall_1':good_score, 'recall_0':bad_score}
