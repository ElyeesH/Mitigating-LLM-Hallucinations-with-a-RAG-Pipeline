from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import evaluate



def load_opt():
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b", torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", use_fast=False)
    return model, tokenizer



