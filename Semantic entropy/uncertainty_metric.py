import json
import os

import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(device)

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

def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # Load JSON content
        return data
    except FileNotFoundError:
        print("Error: File not found!")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format!")
        return None

def semantic_metric():
    sequences = read_json('Llama_TriviaQA.json')
    meteor = evaluate.load('meteor')
    for idx, sample in enumerate(sequences):
        question = sample['question']
        generated_texts = sample['generated_texts']
        id_ = sample["id"]
        result_dict = {}
        unique_generated_texts = list(set(generated_texts))

        answer_list_1 = []
        answer_list_2 = []
        has_semantically_different_answers = False
        inputs = []
        syntactic_similarities = {}
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
        for rouge_type in rouge_types:
            syntactic_similarities[rouge_type] = 0.0

        semantic_set_ids = {}
        for index, answer in enumerate(unique_generated_texts):
            semantic_set_ids[answer] = index

        # print('Number of unique answers:', len(unique_generated_texts))

        if len(unique_generated_texts) > 1:

            # Evalauate semantic similarity
            for i, reference_answer in enumerate(unique_generated_texts):
                for j in range(i + 1, len(unique_generated_texts)): 
                    answer_list_1.append(unique_generated_texts[i])
                    answer_list_2.append(unique_generated_texts[j])

                    qa_1 = question + ' ' + unique_generated_texts[i]
                    qa_2 = question + ' ' + unique_generated_texts[j]

                    input = qa_1 + ' [SEP] ' + qa_2
                    inputs.append(input)
                    encoded_input = tokenizer.encode(input, padding=True)
                    prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
                    predicted_label = torch.argmax(prediction, dim=1)

                    reverse_input = qa_2 + ' [SEP] ' + qa_1
                    encoded_reverse_input = tokenizer.encode(reverse_input, padding=True)
                    reverse_prediction = model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda'))['logits']
                    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

                    if 0 in predicted_label or 0 in reverse_predicted_label:
                        has_semantically_different_answers = True

                    else:
                        semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]
                        
            rouge = evaluate.load('rouge')

            # Evalauate syntactic similarity

            answer_list_1 = []
            answer_list_2 = []
            for i in generated_texts:
                for j in generated_texts:
                    if i != j:
                        answer_list_1.append(i)
                        answer_list_2.append(j)

            results = rouge.compute(predictions=answer_list_1, references=answer_list_2)

            for rouge_type in rouge_types:
                syntactic_similarities[rouge_type] = results[rouge_type]
        result_dict['id'] = id_
        result_dict['semantic'] = {
            'syntactic_similarities': syntactic_similarities,
            'has_semantically_different_answers': has_semantically_different_answers
        }
        list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts]
        result_dict['semantic']['semantic_set_ids'] = list_of_semantic_set_ids
        result_dict['semantic']['semantic_set_ids'] = list_of_semantic_set_ids
        add_calculated_value('Llama_TriviaQA_semantic.json', result_dict)


semantic_metric()
