# Code adapted from https://github.com/alibaba/eigenscore/tree/main
# Original author: Alibaba

import argparse
import glob
import json
import os
import copy
import time

import pandas as pd
import torch
import tqdm
import transformers

import data.nq_open as nq_open
import data.triviaqa as triviaqa
from Models.llama import load_llama
from Models.opt import load_opt
from Models.mistral import load_mistral
from metrics import *
from feature_clip import FeatureClipper
from rag import rag_context


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
parser.add_argument('--num_generations_per_prompt', type=int, default=10)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--decoding_method', type=str, default='greedy')
parser.add_argument('--top_p', type=float, default=0.99)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--seed', type=int, default=2023)



args = parser.parse_args()


def get_dataset_fn(data_name):
    if data_name == 'triviaqa':
        return triviaqa.get_dataset
    if data_name == 'nq_open':
        return nq_open.get_dataset



def get_generation_config(input_ids, tokenizer, data_name):
    assert len(input_ids.shape) == 2
    max_length_of_generated_sequence = 256
    if data_name == 'triviaqa':
        generation_config = triviaqa._generate_config(tokenizer)
    if data_name == 'nq_open':
        generation_config = nq_open._generate_config(tokenizer)
    generation_config['max_new_tokens'] = max_length_of_generated_sequence
    generation_config['early_stopping'] = True
    # https://jaketae.github.io/study/gpt2/#setup
    generation_config['pad_token_id'] = tokenizer.eos_token_id
    return generation_config


@torch.no_grad()
def get_generations(args, seed=1, RAG= False, max_num_gen_once=args.num_generations_per_prompt):
    device = args.device
    model, tokenizer=load_llama()
    model.to(device)
    dataset = get_dataset_fn('triviaqa')(tokenizer)
    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed)['train']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    

    sequences = []
    time_start=time.time()
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if RAG :
            qst=batch['question'][0]
            context=rag_context(qst)
            prompt= nq_open.queston_to_prompt_rag(qst, context)
            inputs = tokenizer(prompt, padding=False, truncation=False)
            input_ids = torch.tensor(inputs['input_ids']).to(device)
            input_ids=input_ids.unsqueeze(0)
            attention_mask = torch.tensor(inputs['attention_mask']).to(device)
            attention_mask=attention_mask.unsqueeze(0)
        else:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
        input_length = input_ids.shape[1]
        generation_config = get_generation_config(input_ids, tokenizer, 'triviaqa')
        generation_config = transformers.GenerationConfig(**generation_config)
        dict_outputs = model.generate(input_ids, attention_mask=attention_mask,
                                        num_beams=1,
                                        do_sample=False,
                                        generation_config=generation_config,
                                        output_hidden_states = True,
                                        return_dict_in_generate=True,
                                        output_scores=True,
                                        )                                                                                

        scores = dict_outputs.scores
        perplexity = get_perplexity_score(scores)
        energy_score = get_energy_score(scores)
        most_likely_generations = dict_outputs.sequences.cpu()[0, input_length:]

        torch.cuda.empty_cache()
        generations = []
        num_gens = args.num_generations_per_prompt
        while num_gens > 0:
            dict_outputs =  model.generate(input_ids, attention_mask=attention_mask,
                            num_beams=1, num_return_sequences=min(max_num_gen_once, num_gens),
                            do_sample=True, top_p=args.top_p, top_k=args.top_k,
                            temperature=args.temperature, generation_config=generation_config,
                            output_hidden_states = True, return_dict_in_generate=True, output_scores=True
                            )

            generation = dict_outputs.sequences[:, input_length:].cpu()
            generations.append(generation)
            num_tokens = get_num_tokens(generation)
            scores = dict_outputs.scores
            predictive_entropy = get_lenghthNormalized_entropy(scores, num_tokens) 
            hidden_states = dict_outputs.hidden_states
            eigenIndicator, eigenValue = getEigenIndicator_v0(hidden_states, num_tokens)
            num_gens -= len(generation)

        generations = torch.nested.nested_tensor(generations).to_padded_tensor(tokenizer.eos_token_id)
        generations = generations.reshape(-1, generations.shape[-1])[:args.num_generations_per_prompt]
        best_generated_text = tokenizer.decode(most_likely_generations, skip_special_tokens=True)
        generated_texts = [tokenizer.decode(s, skip_special_tokens=True) for s in generations]
        lexical_similarity = getLexicalSim(generated_texts)


        curr_seq = dict(
            prompt=tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True),
            id=batch['id'][0],
            question=batch['question'][0],
            answer=batch['answer'][0],
            additional_answers=[],
        )

        curr_seq.update(dict(most_likely_generation=best_generated_text, skip_special_tokens=True), generations=generated_texts,)
        curr_seq.update(
            dict(
                perplexity=perplexity
            )
        )
        curr_seq.update(
            dict(
                energy=energy_score
            )
        )
        curr_seq.update(dict(lexical_similarity=lexical_similarity ))

        curr_seq.update(dict(eigen_score=eigenIndicator))
        #curr_seq['additional_answers'] = [x[0] for x in batch['additional_answers']]  #uncomment if there are additional answers

        sequences.append(curr_seq)
        torch.cuda.empty_cache()
        
        print("Prompt:", tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True))
        print("Question:", batch['question'][0])
        print("AnswerGT:", batch['answer'][0])
        print("MostLikelyAns:", best_generated_text)
        print("Batch_Generations:", generated_texts)
        print("Perplexity:", perplexity)
        print("Energy:", energy_score)
        print("NormalizedEntropy: ", predictive_entropy)
        print("LexicalSimilarity: ", lexical_similarity)
        print("EigenScore: ", eigenIndicator)
    return sequences


def get_num_tokens(generation):
    num_tokens = []
    for ids in generation:
        count = 0
        for id in ids:
            if id>2:
                count+=1
        num_tokens.append(count+1)
    return num_tokens

with open ("llama3.1_triviaqa.json", 'w', encoding='utf-8') as file:
    json.dump(get_generations(args), file, indent=4, ensure_ascii=False)