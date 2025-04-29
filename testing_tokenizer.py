import torch
import torch.nn as nn
import argparse
from model import GPTLanguageModel
from utils import levenshtein_distance
from rouge import Rouge
import tiktoken
import csv
import os

def parse_option():
    parser = argparse.ArgumentParser('argument for testing')

    parser.add_argument('--ckpt', type=str, default='./models/test.pth')
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=6)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_embd', type=int, default=384)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='cuda:0')

    opt = parser.parse_args()

    return opt

def main():
    opt = parse_option()

    # GPT2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s)
    decode = lambda l: enc.decode(l)

    vocab_size = enc.n_vocab  # 50257 for GPT2

    # Load model with correct vocab size
    model = GPTLanguageModel(vocab_size, opt.n_embd, opt.block_size, opt.dropout, opt.device)
    model = model.to(opt.device)
    model.load_state_dict(torch.load(opt.ckpt))

    model.eval()

    total_levenshtein = 0
    total_rouge1 = 0
    total_rouge2 = 0
    total_rougel = 0

    rouge = Rouge()

    num_tests = 16

    for i in range(1, num_tests + 1):
        prompt_path = f'./test3_data/test_prompt_{i}.txt'
        answer_path = f'./test3_data/test_response_{i}.txt'

        with open(prompt_path, 'r', encoding='utf-8') as f:
            text_test_prompt = f.read()

        with open(answer_path, 'r', encoding='utf-8') as f:
            text_test_answer = f.read()

        context = torch.tensor(encode(text_test_prompt), device=opt.device).unsqueeze(0)

        number_gen = len(context[0])
        response = decode(model.generate(context, max_new_tokens=number_gen, block_size=opt.block_size)[0].tolist())

        lev = levenshtein_distance(text_test_answer, response)
        scores = rouge.get_scores(response, text_test_answer)

        total_levenshtein += lev
        total_rouge1 += scores[0]['rouge-1']['f']
        total_rouge2 += scores[0]['rouge-2']['f']
        total_rougel += scores[0]['rouge-l']['f']

    # Average scores
    avg_levenshtein = total_levenshtein / num_tests
    avg_rouge1 = total_rouge1 / num_tests
    avg_rouge2 = total_rouge2 / num_tests
    avg_rougel = total_rougel / num_tests

    print(f"Avg Levenshtein: {avg_levenshtein:.4f}, Avg ROUGE-1: {avg_rouge1:.4f}, Avg ROUGE-2: {avg_rouge2:.4f}, Avg ROUGE-L: {avg_rougel:.4f}")

    # Save results into CSV
    metrics_file = 'metrics_log.csv'

    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Checkpoint', 'Avg Levenshtein', 'Avg ROUGE-1', 'Avg ROUGE-2', 'Avg ROUGE-L'])

    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([opt.ckpt, avg_levenshtein, avg_rouge1, avg_rouge2, avg_rougel])

if __name__ == "__main__":
    main()
