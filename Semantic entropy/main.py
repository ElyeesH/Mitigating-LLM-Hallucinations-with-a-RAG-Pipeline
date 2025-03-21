import json
import functools
import numpy as np
import datasets
import re
import torch
import os
from torch.nn import functional as F


# Use a pipeline as a high-level helper
login = "huggingface_token"

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3").to(device)
Data = datasets.load_dataset("nq_open", split='validation')

def add_calculated_value(file_path, new_value):
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the existing data
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)  # Load existing JSON
            except json.JSONDecodeError:
                data = []  # Initialize as empty list if file is empty or corrupted
    else:
        data = []  # Initialize as empty list if file doesn't exist

    # Append the new value
    data.append(new_value)

    # Write back the updated data
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def get_generation_config(input_ids, tokenizer, data_name):
    max_length_of_generated_sequence = 256
    generation_config = _generate_config(tokenizer)
    generation_config['max_new_tokens'] = max_length_of_generated_sequence
    generation_config['early_stopping'] = True
    generation_config['pad_token_id'] = tokenizer.eos_token_id
    return generation_config

def _generate_config(tokenizer):
    if tokenizer.__class__.__name__ == 'LlamaTokenizer':
        eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', ',', '.']]
    elif tokenizer.__class__.__name__ == 'GPT2Tokenizer':
        eos_token_id = [tokenizer.encode(_)[1] for _ in ['\n', ',', '.']]
    elif tokenizer.__class__.__name__ == "PreTrainedTokenizerFast":
        eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', ',', '.']]
    else:
        raise NotImplementedError
    eos_token_id += [tokenizer.eos_token_id]
    bad_words_ids = [tokenizer(_)['input_ids'] for _ in ['Q:']] 
    return dict(eos_token_id=eos_token_id, bad_words_ids=bad_words_ids)

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
    return f"""Answer only these 6  questions:{get_fs_samples_prompt()}
Q: {sample['question']}
A:"""

def generate():
    results = []
    # print(len(Data))
    for i in range(len(Data)):
        input_text = sample_to_prompt(Data[i])
        inputs = tokenizer(input_text, return_tensors="pt", padding=False, return_attention_mask=True).to(device)
        ans = []
        generation_config = get_generation_config(inputs, tokenizer, 'nq_open')
        generation_config = transformers.GenerationConfig(**generation_config)
        model.eval()
        for j in range(10):
            # Forward pass to calculate loss
            outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"].to(device), generation_config=generation_config, 
                                     do_sample=True, temperature=0.5, top_p=0.99, top_k = 10,  output_hidden_states=True, return_dict_in_generate=True, output_scores=True)

            generated_ids = outputs.sequences  # Extract generated token IDs
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # Find all answers using regex
            answers = re.findall(r"A:\s*(.*)", generated_text[0])

            # Get the last answer
            last_answer = answers[-1] if answers else None
            ans.append(last_answer)
        add_calculated_value('gen.json', {"id": i, "question": Data[i]['question'], "generated_texts": ans})
generate()
