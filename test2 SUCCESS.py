## Demo of torrent of Llama-2

import torch
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

model_name = "meta-llama/Llama-2-70b-chat-hf"  # You can also use "bigscience/bloom" or "bigscience/bloomz"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, add_bos_token=False)
model = AutoDistributedModelForCausalLM.from_pretrained(model_name).cuda()
# Embeddings & prompts are on your device, transformer blocks are distributed across the Internet

inputs = tokenizer("A ruffled old cat reclined lazily on the weathered lower branch of an old Maple tree, licking it's paw and", return_tensors="pt")["input_ids"].cuda()
outputs = model.generate(inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))  # will complete the above sentence up to max_new_tokens