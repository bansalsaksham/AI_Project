# gpt-domain-finetune

> **Spring 2025 · ECE2806 AI Foundations · Georgia Institute of Technology**  
> *Saksham Bansal · sakshambansal@gatech.edu*

Fine-tuning a GPT-style transformer from scratch on a curated academic corpus covering **Principal Component Analysis (PCA)**, **Reinforcement Learning (RL)**, and **Decision Trees (DT)** — with systematic hyperparameter sweeps evaluated via ROUGE and Levenshtein distance.

---

## Motivation

Generic LLMs trained on web-scale data lack technical depth in niche academic domains — they misuse terminology and fail to produce structured reasoning. This project demonstrates that **focused fine-tuning on a small, curated dataset** can meaningfully improve domain-specific coherence and accuracy, without scaling model size.

---

## Methodology

### Pipeline

```
Dataset Preparation → Tokenization Strategy → Model Architecture → Hyperparameter Sweep → Evaluation
```

### Dataset
A custom corpus (`train_data/AI2806.txt`) was hand-curated from AI Foundations course material on PCA, RL, and DT, reviewed for clarity and consistency.

### Model Architecture
GPT-style causal transformer with:
- Causal self-attention
- Positional embeddings
- Stacked transformer encoder blocks
- Final projection to vocabulary space

**Baseline:** 6 attention heads, 6 transformer layers.

### Hyperparameter Sweeps

One parameter was varied per experiment, keeping all others fixed.

| Hyperparameter | Values Tested |
|---|---|
| Learning Rate | 0.03, 0.003, 0.0003, 0.00003 |
| Batch Size | 64, 128 |
| Max Iterations | 500, 1000, 2000, 3000, 4000, 5000, 10000 |
| Attention Heads | 4, 6, 8 |
| Transformer Layers | 4, 6, 8 |
| Tokenizer | Basic Character-Level, GPT-2 |

### Evaluation
Models tested on **16 standardized prompt–answer pairs** using:
- **Levenshtein Distance** — character-level edit distance (lower = better)
- **ROUGE-1** — unigram word overlap
- **ROUGE-2** — bigram phrase overlap
- **ROUGE-L** — longest common subsequence

---

## Key Results

| Finding | Detail |
|---|---|
| Best Learning Rate | **0.003** → lowest Levenshtein (~81.75), highest ROUGE-1 (~0.186) |
| Best Iteration Count | **~3000** → peak ROUGE-1; gains flatten and overfit beyond 5000 |
| Batch Size | **128** → better edit distance; **64** → better ROUGE scores |
| Attention Heads | **8 heads** best overall ROUGE-1; gains diminish past 6 |
| Transformer Layers | **4–6 layers** best balance; 8 layers added cost without gain |
| Tokenizer | **GPT-2** outperformed basic character-level encoding |

### Best Overall Configuration
```
Learning Rate:    0.003
Batch Size:       128
Max Iterations:   2000–3000
Attention Heads:  8
Layers:           4–6
Tokenizer:        GPT-2
```

---

## Project Structure

```
gpt-domain-finetune/
├── training_base.py          # Train with basic character-level tokenizer
├── training_tokenizer.py     # Train with GPT-2 tokenizer
├── testing_base.py           # Evaluate basic tokenizer model
├── testing_tokenizer.py      # Evaluate GPT-2 tokenizer model
├── generate_examples.py      # Output model-generated answers
├── run_script_final          # Bash script to automate all experiments
├── model.py                  # GPT architecture definition
├── utils.py                  # Encoding, batching, distance utilities
├── plot_metrics.py           # Plot evaluation metrics from log
├── metrics_log.csv           # All experiment results
├── requirements.txt          # Python dependencies
├── plots/                    # Evaluation graphs
├── train_data/               # Custom training corpus (AI2806.txt)
└── test_data/                # test_prompt_#.txt + test_response_#.txt
```

> **Note:** Trained model checkpoints (`.pth` files) are not included due to file size. After training, they will be saved locally to a `models/` directory.

---

## Setup & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/bansalsaksham/gpt-domain-finetune
cd gpt-domain-finetune
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

**Basic tokenizer:**
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

**GPT-2 tokenizer:**
```bash
python3 training_tokenizer.py \
  --training_file train_data/AI2806.txt \
  --tokenization_strategy GPT2 \
  --save_file models/model_tokenizer_gpt2.pth
```

### 4. Evaluate

```bash
python3 testing_base.py --ckpt models/model_lr_0.003.pth
python3 testing_tokenizer.py --ckpt models/model_tokenizer_gpt2.pth
```

### 5. Generate Predictions

```bash
python3 generate_examples.py --ckpt models/model_lr_0.003.pth
```

### 6. Run All Experiments (Automated)

```bash
bash run_script_final
```

---

## Full Metrics Table

| Checkpoint | Avg Levenshtein | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|
| model_lr_0.03 | 82.125 | 0.1475 | 0.0278 | 0.1241 |
| **model_lr_0.003** | **81.750** | **0.1860** | **0.0377** | **0.1518** |
| model_lr_0.0003 | 82.375 | 0.1497 | 0.0251 | 0.1148 |
| model_lr_0.00003 | 83.375 | 0.1615 | 0.0254 | 0.1244 |
| model_bs_64 | 84.125 | 0.1689 | 0.0306 | 0.1269 |
| model_bs_128 | 81.375 | 0.1633 | 0.0250 | 0.1246 |
| model_iters_1000 | 80.8125 | 0.1769 | 0.0252 | 0.1268 |
| model_iters_3000 | 81.375 | 0.1781 | 0.0381 | 0.1357 |
| model_iters_10000 | 82.000 | 0.1650 | 0.0266 | 0.1262 |
| model_heads_8 | 81.8125 | 0.1782 | 0.0255 | 0.1450 |
| model_layers_4 | 82.3125 | 0.1694 | 0.0325 | 0.1303 |
| model_tokenizer_gpt2 | 83.0625 | 0.1580 | 0.0253 | 0.1284 |

---

## Challenges

- **CUDA memory limits** — batch sizes above 128 caused out-of-memory errors.
- **Learning rate sensitivity** — small changes had large effects on convergence stability.
- **Dataset size** — limited corpus restricted generalization capacity.
- **Metric fragility** — ROUGE and Levenshtein don't fully capture semantic correctness.
- **Training time** — runs up to 10,000 iterations required careful GPU scheduling.

---

## References

1. Vaswani et al., "Attention Is All You Need", NeurIPS 2017
2. Wolf et al., "Transformers: State-of-the-Art NLP", EMNLP 2020
3. Lin, "ROUGE: A Package for Automatic Evaluation of Summaries", 2004
4. OpenAI, "GPT-2: Better Language Models and Their Implications", 2019
5. PyTorch Documentation — https://pytorch.org/docs/stable/index.html

---

## Acknowledgements

Built upon the Spring 2025 AIFirst Project Starter Code · ECE2806 · Georgia Institute of Technology.
