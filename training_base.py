import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTLanguageModel
import argparse
from utils import get_batch, estimate_loss, levenshtein_distance
import string
import tiktoken

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--block_size', type=int, default=256,
                        help='Size of blocks to process vocabulary')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout')
    parser.add_argument('--n_heads', type=int, default=6, help='Number of Heads in Attention Block')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of Layers in Attention Block')
    parser.add_argument('--n_embd', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--loss', type=str, default='NLL')
    parser.add_argument('--model', type=str, default='basic')
    parser.add_argument('--training_file', type=str, default='./train_data/AI2806.txt')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--save_file', type=str, default='./models/test.pth')
    parser.add_argument('--device', type=str, default='cuda:0')

    opt = parser.parse_args()

    return opt

def main():
    opt = parse_option()

    # Load training data
    with open(opt.training_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # --- Use GPT2 tokenizer ---
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s)
    decode = lambda l: enc.decode(l)

    vocab_size = enc.n_vocab  # 50257 for GPT2

    # Encode the entire dataset
    data = torch.tensor(encode(text), dtype=torch.long)

    # Split into train and validation
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Create model
    model = GPTLanguageModel(vocab_size, opt.n_embd, opt.block_size, opt.dropout, opt.device)
    model = model.to(opt.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    step = 0
    while step < opt.max_iters:
        model.train()

        idx = torch.randint(0, train_data.size(0) - opt.block_size, (opt.batch_size,))
        x = torch.stack([train_data[i:i+opt.block_size] for i in idx])
        y = torch.stack([train_data[i+1:i+1+opt.block_size] for i in idx])

        x, y = x.to(opt.device), y.to(opt.device)

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            model.eval()
            idx = torch.randint(0, val_data.size(0) - opt.block_size, (opt.batch_size,))
            x_val = torch.stack([val_data[i:i+opt.block_size] for i in idx]).to(opt.device)
            y_val = torch.stack([val_data[i+1:i+1+opt.block_size] for i in idx]).to(opt.device)

            with torch.no_grad():
                logits_val, val_loss = model(x_val, y_val)

            print(f"step {step}: train loss {loss.item():.4f}, val loss {val_loss.item():.4f}")

        step += 1

    torch.save(model.state_dict(), opt.save_file)

if __name__ == "__main__":
    main()
