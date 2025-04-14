import pickle
from conllu_to_data import TokenSpan, TrainingDatum
from datasets import Dataset, DatasetDict


phrase2name = {
    'NP': ('grup nominal', 'M'),
    'VP': ('grup verbal', 'M'),
    'AdjP' : ('grup adjectival', 'M'),
    'AdvP': ('grup adverbial', 'M'),
    'PP': ('grup prepozițional', 'M'),
    'CP': ('subordonate conjuncționale', 'F'),
    'RP': ('subordonate relative', 'F'),
}
def extract_phrase_head(td : TrainingDatum, include_phrase_type : bool = True) -> dict:
    if not td.head:
        return {}
    context = td.constituent
    answer = td.head
    answer_index = context.find(answer)
    assert answer_index >= 0
    question = 'Care este centrul'
    if td.label in phrase2name and include_phrase_type:
        phrase, gen = phrase2name[td.label]
        question += f" {'acestui' if gen == 'M' else 'acestei'} {phrase}?"
    else:
        question += ' acestui grup sintactic?'
    return {'context': context, 'question':question,
            'answers': {'answer_start': [answer_index], 'text':[answer]}}

def extract_subunits(td : TrainingDatum) -> dict:
    if not td.subunits:
        return {}
    context = td.constituent
    question = 'În ce părți poate fi împărțit acest constituent?'
    ans_texts, answ_starts = [], []
    for subunit in td.subunits:
        ans_texts.append(subunit)
        answ_starts.append(context.find(subunit))
    return {'context': context, 'question':question,
            'answers': {'answer_start': answ_starts, 'text':ans_texts}}


if __name__ == "__main__":
    with open('./datasets/raw_data-dev-DICT.p', 'rb') as handle:
        raw_training_dev = pickle.load(handle)

    raw_training_dev = [TrainingDatum(**d) for d in raw_training_dev]

    phrase_parts_qa = [extract_phrase_head(td) for td in raw_training_dev]
    phrase_parts_qa = [d for d in phrase_parts_qa if d]
    ds = Dataset.from_list(phrase_parts_qa)
    ds_dict = ds.train_test_split(0.25)
    ds_dict.save_to_disk('./datasets/dsdict_qa_parts-dev-1')

