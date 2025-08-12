import torch

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer
)

import datasets

model_source = 'dumitrescustefan/t5-v1_1-base-romanian'

ENCODER_LEN = 512
DECODER_LEN = 256

