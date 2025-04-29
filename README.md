# Spring 2025 AI Foundations Project – Fine-Tuning GPT for PCA, Reinforcement Learning, and Decision Trees

**Author:** Saksham Bansal  
**Course:** ECE2806 – AI Foundations  
**Repository Type:** Educational Research – AI Customization  

---

## Project Overview

This project demonstrates how a **custom fine-tuned GPT model** can be adapted to better understand and respond to domain-specific topics in:

- Principal Component Analysis (PCA)  
- Reinforcement Learning (RL)  
- Decision Trees (DT)  

These are core topics from the ECE2806 course curriculum.

Unlike generic pretrained LLMs, this project emphasizes:

- Preparing a **focused academic dataset**
- Training a GPT model from scratch
- Exploring key **hyperparameter effects**
- Evaluating model performance using structured metrics

---

## Project Setup – AI Makerspace Instructions

> Follow these terminal steps to clone, train, test, and evaluate your model.

### 1. Clone the Repository

```bash
git clone https://github.gatech.edu/sbansal91/ECE2806_Final_Project
cd ECE2806_Final_Project
```

### 2. Set Up Environment

Install dependencies manually:
```bash
pip install -r requirements.txt
```

---

## Folder & File Structure

| File / Folder               | Description |
|----------------------------|-------------|
| `training_base.py`         | Trains model using basic tokenizer |
| `training_tokenizer.py`    | Trains model using GPT-2 tokenizer |
| `testing_base.py`          | Evaluates model (Levenshtein, ROUGE) |
| `testing_tokenizer.py`     | Evaluates GPT-2 tokenizer model |
| `generate_examples.py`     | Outputs model-generated answers |
| `run_script_final`         | Bash script to automate experiments |
| `model.py`                 | GPT architecture definition |
| `utils.py`                 | Utilities: encoding, batching, distances |
| `plot_metrics.py`          | Plots evaluation metrics from log |
| `metrics_log.csv`          | Stores all experiment results |
| `requirements.txt`         | Required Python libraries |
| `models/`                  | Saved model weights (.pth) |
| `plots/`                   | Evaluation graphs |
| `train_data/AI2806.txt`    | Custom training corpus (PCA, RL, DT) |
| `test3_data/`              | Contains test_prompt_#.txt, test_response_#.txt |

---

## How to Run the Project

### A. Train the Model

Basic tokenizer model:
```bash
python3 training_base.py \
  --training_file train_data/AI2806.txt \
  --learning_rate 0.003 \
  --batch_size 64 \
  --n_heads 6 \
  --n_layer 6 \
  --max_iters 2000 \
  --save_file models/model_lr_0.003.pth
```

GPT-2 tokenizer model:
```bash
python3 training_tokenizer.py \
  --training_file train_data/AI2806.txt \
  --tokenization_strategy GPT2 \
  --save_file models/model_tokenizer_gpt2.pth
```

---

### B. Test the Model

Basic model:
```bash
python3 testing_base.py --ckpt models/model_lr_0.003.pth
```

GPT-2 tokenizer model:
```bash
python3 testing_tokenizer.py --ckpt models/model_tokenizer_gpt2.pth
```

---

### C. Automate All Experiments

```bash
bash run_script_final
```

---

## Evaluation Metrics

The model is tested on **16 standardized prompt–answer pairs**.

| Metric               | Meaning |
|----------------------|---------|
| Levenshtein Distance | Character-level edit distance |
| ROUGE-1              | Word overlap (unigrams) |
| ROUGE-2              | Phrase overlap (bigrams) |
| ROUGE-L              | Longest common subsequence |

Results are logged in `metrics_log.csv` and visualized via `plot_metrics.py`.

---

## Output Example

Generate model predictions:

```bash
python3 generate_examples.py --ckpt models/model_lr_0.003.pth
```

Sample output:
```
=== Example 1 ===
Prompt: What is Principal Component Analysis (PCA)?
Model Output: What is PCA? (effects= The...
Ground Truth: PCA reduces data dimensionality while preserving variance.
```

---

## Summary of Features

- Full control over model architecture: layers, heads, embedding size  
- Detailed logging of all experimental settings and outcomes  
- Support for multiple tokenization strategies (Basic, GPT-2)  
- Scripted automation for reproducibility  

---

## Acknowledgements

This repository builds upon the **Spring 2025 AIFirst Project Starter Code** and is inspired by Andrej Karpathy's *zero-to-hero* LLM training framework.

All training, testing, and evaluation was conducted from scratch to analyze hyperparameter effects on specialization in academic domains.
