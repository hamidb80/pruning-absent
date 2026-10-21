import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_pm(val):
    """Convert '0.0665±0.0674' -> (0.0665, 0.0674)"""
    if isinstance(val, (int, float)):
        return val, 0.0
    try:
        mean, std = val.split('±')
        return float(mean), float(std)
    except (ValueError, AttributeError):
        return float(val), 0.0

# --- 1. Data Processing (unchanged) ---
df = pd.read_csv("./results/summary.csv")
base_cols = ['random_activation', 'most_activation', 'least_activation']

df_melted = df.melt(
    id_vars=['method', 'target', 'sparsity'],
    value_vars=base_cols,
    var_name='metric',
    value_name='raw_value'
)

df_melted[['value_mean', 'value_std']] = pd.DataFrame(
    df_melted['raw_value'].apply(parse_pm).tolist(),
    index=df_melted.index
)

df_final = df_melted.drop(columns=['raw_value'])

# --- 2. Plot all combinations in a single row ---
methods = df_final['method'].unique()
targets = df_final['target'].unique()

# Create a list of all (method, target) combinations
combinations = [(m, t) for m in methods for t in targets]

# Determine how many subplots we need
n_cols = len(combinations)
fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)
axes = axes.flatten()  # easy indexing

for ax, (m, t) in zip(axes, combinations):
    vdf = df_final[
        (df_final['method'] == m) &
        (df_final['target'] == t) &
        (df_final['sparsity'] <= 0.9)
    ]

    if len(vdf) == 0:
        ax.set_visible(False)  # hide empty subplot
        continue

    metrics = vdf['metric'].unique()
    colors = sns.color_palette(n_colors=len(metrics))

    sns.lineplot(
        data=vdf,
        x='sparsity',
        y='value_mean',
        hue='metric',
        palette=colors,
        marker='o',
        ax=ax
    )
    ax.set_title(f"pruning: {m} | target: {t}")
    ax.set_ylabel('value')
    
    for metric_val, color in zip(metrics, colors):
        line_data = vdf[vdf['metric'] == metric_val].sort_values('sparsity')
        ax.fill_between(
            x=line_data['sparsity'],
            y1=line_data['value_mean'] - line_data['value_std'],
            y2=line_data['value_mean'] + line_data['value_std'],
            color=color,
            alpha=0.2
        )

# If some subplots were hidden, adjust layout
plt.tight_layout()
os.makedirs('./build', exist_ok=True)
plt.savefig('./build/all_plots_row.png', bbox_inches='tight')
plt.close()