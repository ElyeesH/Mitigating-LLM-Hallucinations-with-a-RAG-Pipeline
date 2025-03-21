# Code adapted from https://github.com/alibaba/eigenscore/tree/main
# Original author: Alibaba

import functools
import os
import pathlib
import pickle

# import config
import datasets
import ipdb
import numpy as np
import pandas as pd
import torch
from rag import rag_context
import time


@functools.lru_cache()
def get_fs_samples_prompt():
    data = datasets.load_dataset("nq_open", split='train')
    indices = np.random.RandomState(42).choice(len(data), 5)
    ret = ''
    for i in indices:
        i = int(i)
        ret += '\nQ: ' + data[i]['question'] + '\nA: ' + data[i]['answer'][0]
    return ret

def sample_to_prompt(sample, **kwargs):
    if isinstance(sample['question'], list):
        return [sample_to_prompt({'question': _}, **kwargs) for _ in sample['question']]
    return f"""Answer these questions:{get_fs_samples_prompt()}
Q: {sample['question']}
A:"""


def queston_to_prompt_rag(question, context, **kwargs):
    return f"""Answer the last question based on the provided context and in the same format of the following examples:{get_fs_samples_prompt()}
Context: {context}
Q: {question}
A:"""

def _generate_config(tokenizer):
    if tokenizer.__class__.__name__ == 'LlamaTokenizer':
        eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', '.']]
    elif tokenizer.__class__.__name__ == 'GPT2Tokenizer':
        eos_token_id = [tokenizer.encode(_)[1] for _ in ['\n', '.']]
    elif tokenizer.__class__.__name__ == "PreTrainedTokenizerFast":
        eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', '.']]
    else:
        raise NotImplementedError
    eos_token_id += [tokenizer.eos_token_id]
    bad_words_ids = [tokenizer(_)['input_ids'] for _ in ['Q:']] 
    return dict(eos_token_id=eos_token_id, bad_words_ids=bad_words_ids)


def get_dataset(tokenizer, start=0, end=3610):
    # For Natural Questions we use the test split used for open-domain question answering containing 3610 questions.
    data = datasets.load_dataset("nq_open", split='validation')
    data=data.select(range(start,end))
    id_map = {_['question']:str(i) for i, _ in enumerate(data)}

    def process_instance(example):
        example['id'] = id_map[example['question']]
        all_answers = example.pop('answer')
        example['additional_answers'] = all_answers[1:]
        example['answer'] = all_answers[0]

        example['prompt'] = sample_to_prompt({k:example[k] for k in ['question']})
        inputs = tokenizer(example['prompt'], padding=False, truncation=False)
        outputs = tokenizer(all_answers[0], padding=False, truncation=False)
        example['input_ids'] = inputs['input_ids']
        example["attention_mask"] = inputs.attention_mask
        example["labels"] = outputs.input_ids.copy()
        example["labels"] = [-100 if _ == tokenizer.pad_token_id else _ for _ in example["labels"]]
        return example
    data = data.map(process_instance, load_from_cache_file=False)
    data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
        output_all_columns=True)
    return data


