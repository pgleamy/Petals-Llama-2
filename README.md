# Llama-2-70b Chat Model

This repository contains the code for generating text using the Llama-2-70b chat model. The model is distributed across the internet, with embeddings and prompts on your device and transformer blocks distributed across the Internet.

## Requirements

- Python 3.6 or later
- PyTorch 1.8.0 or later
- Transformers library
- Petals library

## Installation

You can install the required packages using pip:

```bash
pip install torch transformers petals
```

## Usage

Here is a simple example of how to use the model:

```python
import torch
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

model_name = "meta-llama/Llama-2-70b-chat-hf"  
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, add_bos_token=False)
model = AutoDistributedModelForCausalLM.from_pretrained(model_name).cuda()

inputs = tokenizer("A ruffled old cat reclined lazily on the weathered lower branch of an old Maple tree, licking it's paw and", return_tensors="pt")["input_ids"].cuda()
outputs = model.generate(inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))  # will complete the above sentence up to max_new_tokens
```

In this example, the model will complete the sentence "A ruffled old cat reclined lazily on the weathered lower branch of an old Maple tree, licking it's paw and" with up to 100 new tokens.

## Model

The model used in this example is "meta-llama/Llama-2-70b-chat-hf". You can also use "bigscience/bloom" or "bigscience/bloomz" as the model.

## Tokenizer

The tokenizer used in this example is from the transformers library. It is used to convert the input text into a format that the model can understand.

## Generating Text

The `model.generate()` function is used to generate the text. The `max_new_tokens` parameter determines the maximum number of new tokens to generate.

## Output

The output of the model is decoded using the `tokenizer.decode()` function and printed to the console.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.