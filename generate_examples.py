import torch
import tiktoken
from model import GPTLanguageModel
import argparse
import os

# Set model and device
model_ckpt = 'models/model_lr_0.003.pth'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# GPT-2 Tokenizer
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)

# Load model
vocab_size = enc.n_vocab
block_size = 256 
n_embd = 384
n_heads = 6
n_layers = 6
dropout = 0.2

model = GPTLanguageModel(vocab_size, n_embd, block_size, dropout, device)
model.load_state_dict(torch.load(model_ckpt))
model = model.to(device)
model.eval()

# Choose prompts you want to display (1, 5, 10 for example)
prompt_indices = [1, 2, 3, 4, 6, 7]

for idx in prompt_indices:
    prompt_path = f'test3_data/test_prompt_{idx}.txt'
    response_path = f'test3_data/test_response_{idx}.txt'

    if not os.path.exists(prompt_path) or not os.path.exists(response_path):
        continue

    # Load prompt
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_text = f.read()

    # Load ground truth
    with open(response_path, 'r', encoding='utf-8') as f:
        ground_truth = f.read()

    # Generate model response
    context = torch.tensor(encode(prompt_text), device=device).unsqueeze(0)
    number_gen = len(context[0])
    model_output = decode(model.generate(context, max_new_tokens=number_gen, block_size=block_size)[0].tolist())

    # Display nicely
    print(f"\n=== Example {idx} ===")
    print(f"Prompt:\n{prompt_text.strip()}\n")
    print(f"Model Output:\n{model_output.strip()}\n")
    print(f"Ground Truth:\n{ground_truth.strip()}\n")
    print("="*80)
