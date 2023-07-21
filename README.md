# Llama-2-70b Chat Model

This repository contains the code for generating text using the Llama-2-70b chat model using the Petals library. Via Petals, models not otherwise useable on weaker consumer graphics cards are supported. Models are distributed across the internet, with embeddings and prompts on your device and transformer blocks distributed across the Internet.

If you are under Windows, like me, you need to run this under a WSL2 environment (Windows Subsystem for Linus, with Ubuntu installed). Some dependencies required by Petals are not compatible with Windows. Install this in a virtual environment. Installing Petals is a slow process, and downloading Llama-2-70b-chat-hf on the first execution is 18.5GB.

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

The model used in this example is "meta-llama/Llama-2-70b-chat-hf". You can also use "bigscience/bloom" or "bigscience/bloomz" as the model. I ran Llama-2-70b-chat-hf on my RTX 3060 12GB. It only works as a result of Petals.
Petals v2.0 was released today or yesterday (July 20th) so my code will still work but may need to be updated.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.