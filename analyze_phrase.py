import string

import datasets

task = 'question-answering'
model_source = 'hartular/head_dependent-qa-bbert-dev'
dataset_src = 'hartular/dsdict_qa_head_dependent-bare-question-dev'

print('Importing libraries...')

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from transformers.pipelines.question_answering import QuestionAnsweringPipeline

print('Importing models')

model = AutoModelForQuestionAnswering.from_pretrained(model_source)
tokenizer = AutoTokenizer.from_pretrained(model_source)
mpipe = pipeline(task, model=model, tokenizer=tokenizer)

TO_STRIP = ''.join(set(string.punctuation + string.whitespace).difference('[](){}'))

def analyze_phrase(text : str, qa_pipe : QuestionAnsweringPipeline, q_with_center : bool = True) -> (str, list[str]):
    center = qa_pipe(context=text, question='Care este centrul?')['answer']
    dependents = []
    while len(text) > len(center):
        question = 'Care este primul dependent' + (f' al centrului "{center}"' if q_with_center else '') + '?'
        dependent = qa_pipe(context=text, question=question)['answer']
        assert dependent in text
        text = text.replace(dependent, '')
        text = text.strip(TO_STRIP)
        dependent = dependent.strip(TO_STRIP)
        dependents.append(dependent)
    return center, dependents

def compare_answers(question_dict : dict, qa_pipe : QuestionAnsweringPipeline) -> int:
    answer_dict = qa_pipe('question-answering',
                          context=question_dict['question'],
                          question=question_dict['question'])
    correct_answer = question_dict['answers']['text'][0] if question_dict['answers']['text'] else ''
    return int(answer_dict['answer'] == correct_answer)


from datasets import Dataset, DatasetDict


dsd = datasets.load_dataset(dataset_src)