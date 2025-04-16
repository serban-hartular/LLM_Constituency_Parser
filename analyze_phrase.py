import string

task = 'question-answering'
model_source = 'hartular/head_dependent-qa-bbert-dev'

print('Importing libraries...')

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from transformers.pipelines.question_answering import QuestionAnsweringPipeline

print('Importing models')

model = AutoModelForQuestionAnswering.from_pretrained(model_source)
tokenizer = AutoTokenizer.from_pretrained(model_source)
mpipe = pipeline(task, model=model, tokenizer=tokenizer)

TO_STRIP = ''.join(set(string.punctuation + string.whitespace).difference('[](){}'))

def analyze_phrase(text : str, qa_pipe : QuestionAnsweringPipeline) -> (str, list[str]):
    center = qa_pipe(context=text, question='Care este centrul?')['answer']
    dependents = []
    while len(text) > len(center):
        dependent = qa_pipe(context=text, question=f'Care este primul dependent al centrului "{center}"?')['answer']
        assert dependent in text
        text = text.replace(dependent, '')
        text = text.strip(TO_STRIP)
        dependent = dependent.strip(TO_STRIP)
        dependents.append(dependent)
    return center, dependents


