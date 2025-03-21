from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

def load_mistral():
    tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return model, tokenizer
