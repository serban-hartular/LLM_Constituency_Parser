import itertools
import pickle
from conllu_to_data import TokenSpan, TrainingDatum
from datasets import Dataset, DatasetDict, DatasetInfo


phrase2name = {
    'NP': ('grup nominal', 'M'),
    'VP': ('grup verbal', 'M'),
    'AdjP' : ('grup adjectival', 'M'),
    'AdvP': ('grup adverbial', 'M'),
    'PP': ('grup prepozițional', 'M'),
    'CP': ('subordonate conjuncționale', 'F'),
    'RP': ('subordonate relative', 'F'),
}
def phrase_head_question(td : TrainingDatum, include_phrase_type : bool = True) -> list[dict]:
    if not td.head or len(td.subunits) < 2:
        return []
    context = td.constituent
    answer = td.head
    answer_index = context.find(answer)
    assert answer_index >= 0
    question = 'Care este centrul?'
    return [{'context': context, 'question':question,
            'answers': {'answer_start': [answer_index], 'text':[answer]}}]

def subunit_questions(td : TrainingDatum) -> list[dict]:
    if not td.subunits or len(td.subunits) < 2:
        return []
    question = 'Care este primul dependent?'
    subunits = list(td.subunits)
    dependents = [d for d in subunits if d != td.head]
    question_list = []
    while dependents:
        context = ' '.join(subunits)
        answer = dependents[0]
        answer_index = context.index(answer)
        question_list.append({'context':context, 'question':question,
                              'answers':{'answer_start': [answer_index], 'text':[answer]}})
        subunits.remove(answer)
        dependents = dependents[1:]
    return question_list

if __name__ == "__main__":
    with open('./datasets/raw_data-dev-DICT.p', 'rb') as handle:
        raw_training_dev = pickle.load(handle)

    raw_training_dev = [TrainingDatum(**d) for d in raw_training_dev]

    phrase_parts_qa = list(itertools.chain.from_iterable([phrase_head_question(td) + subunit_questions(td) for td in raw_training_dev]))
    ds = Dataset.from_list(phrase_parts_qa)
    ds.info.description = """This is a qa dataset for determining the head and dependents of a syntactic group.
    When asking for dependents, the question does not include the head! Based on RoRefTrees."""

    ds_dict = ds.train_test_split(0.25)
    ds_dict.save_to_disk('./datasets/dsdict_qa_head_dependent-bare-question-dev')

