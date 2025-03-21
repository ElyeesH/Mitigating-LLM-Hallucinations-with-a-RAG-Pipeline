import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from huggingface_hub import login

login(token="huggingface_token")
# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


llh_shift = torch.tensor(5.0)

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

def get_overall_log_likelihoods(list_of_results):
    result_dict = {}

    list_of_keys = ['average_neg_log_likelihoods']
    dicto = read_json("Llama_TriviaQA_semantic.json")
    similarities_dict = {}
    for elem in dicto:
        # print(elem)
        similarities_dict[elem['id']] = {'semantic_set_ids': elem["semantic"]["semantic_set_ids"]}

    for key in list_of_keys:
        list_of_ids = []
        overall_results = []
        result_dict['semantic_set_ids'] = []
        for idx, sample in list_of_results.items():
            average_neg_log_likelihoods = sample[key]
            list_of_ids.append(idx)
            overall_results.append(average_neg_log_likelihoods)
            result_dict['semantic_set_ids'].append(torch.tensor(similarities_dict[idx]['semantic_set_ids'], device=device))

        result_dict[key] = torch.stack(overall_results)
        result_dict['semantic_set_ids'] = torch.stack(result_dict['semantic_set_ids'])
        result_dict['id'] = list_of_ids
    return result_dict

def get_predictive_entropy_over_concepts(log_likelihoods, semantic_set_ids, list_of_ids):
    # Compute the semantic entrop
    # This is ok because all the models have the same semantic set ids
    entropies = []
    print(log_likelihoods.shape[0], semantic_set_ids.shape[0])
    for row_index in range(log_likelihoods.shape[0]):
        aggregated_likelihoods = []
        row = log_likelihoods[row_index]
        semantic_set_ids_row = semantic_set_ids[row_index]
        semantic_set_ids_row = semantic_set_ids_row.to(row.device)
        for semantic_set_id in torch.unique(semantic_set_ids_row):
            aggregated_likelihoods.append(torch.logsumexp(row[semantic_set_ids_row == semantic_set_id], dim=0))
        aggregated_likelihoods = torch.stack(aggregated_likelihoods) - llh_shift
        entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0])
        entropies.append((entropy, list_of_ids[row_index]))
    return entropies

def log():
    sequences = read_json("Llama_TriviaQA.json")
    result = {}
    for sample in sequences:
        result_dict = {}
        prompt = sample['question']
        generations = sample['generated_texts']
        id_ = sample['id']

        average_neg_log_likelihoods = torch.zeros((len(generations),))
        for generation_index in range(len(generations)):

            generation = generations[generation_index]
            combined_input = prompt  + " " + generation
            # Tokenize the input sequence
            encode = tokenizer(prompt, return_tensors = 'pt')
            encodings = tokenizer(combined_input, return_tensors="pt")
            input_ids = encodings.input_ids.to(device)
            seq_len = encode.input_ids.size(1)

            target_ids = input_ids.clone()
            target_ids[:, :seq_len] = -100

            with torch.no_grad():
                # Model forward pass
                outputs = model(input_ids, labels=target_ids)
                average_neg_log_likelihoods[generation_index] = outputs.loss
        result_dict['average_neg_log_likelihoods'] = average_neg_log_likelihoods
        result[id_] = result_dict
    overall_results = get_overall_log_likelihoods(result)

    predictive_entropy_over_concepts = get_predictive_entropy_over_concepts(-overall_results['average_neg_log_likelihoods'],
                                                                        overall_results['semantic_set_ids'], overall_results['id'])
    print(predictive_entropy_over_concepts)
    for entro, idx in predictive_entropy_over_concepts:
        result_dict = {}
        result_dict['id'] = idx
        result_dict['question'] = sequences[idx]['question']
        result_dict['semantic_entropy'] = entro.item()
        add_calculated_value('Llama_TriviaQA_entropy.json', result_dict) 


log()
