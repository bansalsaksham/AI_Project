import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the metrics CSV
metrics_file = 'metrics_log.csv'

if not os.path.exists(metrics_file):
    print("metrics_log.csv not found. Run your training/testing first.")
    exit(1)

df = pd.read_csv(metrics_file)

# Function to extract hyperparameters from filenames
def extract_param(filename, keyword):
    if keyword in filename:
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part == keyword and i + 1 < len(parts):
                return parts[i+1].replace('.pth', '')
    return None

df['Learning Rate'] = df['Checkpoint'].apply(lambda x: extract_param(x, 'lr'))
df['Batch Size'] = df['Checkpoint'].apply(lambda x: extract_param(x, 'bs'))
df['Max Iters'] = df['Checkpoint'].apply(lambda x: extract_param(x, 'iters'))
df['Heads'] = df['Checkpoint'].apply(lambda x: extract_param(x, 'heads'))
df['Layers'] = df['Checkpoint'].apply(lambda x: extract_param(x, 'layers'))
df['Tokenizer'] = df['Checkpoint'].apply(lambda x: 'GPT2' if 'tokenizer' in x else 'ASCII')

# Make directory to save plots
os.makedirs('plots', exist_ok=True)

# Plotting function
def plot_metric(x_param, metric_name):
    plt.figure(figsize=(8,6))
    filtered = df.dropna(subset=[x_param])
    if filtered.empty:
        return
    x = filtered[x_param].astype(float)
    y = filtered[metric_name]
    plt.plot(x, y, marker='o')
    plt.xlabel(x_param)
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs {x_param}')
    plt.grid(True)
    plt.savefig(f'plots/{metric_name}_vs_{x_param}.png')
    plt.close()

# Metrics to plot
metrics = ['Avg Levenshtein', 'Avg ROUGE-1', 'Avg ROUGE-2', 'Avg ROUGE-L']

# Parameters to plot against
params = ['Learning Rate', 'Batch Size', 'Max Iters', 'Heads', 'Layers']

# Generate all plots
for metric in metrics:
    for param in params:
        plot_metric(param, metric)

print("✅ All plots saved in 'plots/' directory.")
s