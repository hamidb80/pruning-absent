import os
import numpy as np
import torch
from torchvision import datasets, transforms, models

# CONFIG --------------------

CPU = 'cpu'
CUDA = 'cuda'
DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)

data_dir    = 'D:/sautkin.imagenet1kvalid/'
class_a     = "00104"
class_b     = "00190"
num_alphas  = 11
num_pairs   = 5           # <-- how many A/B image pairs to try
pair_seed   = 0           # <-- fix for reproducible pair selection
model_name  = "resnet50"
data_ratio  = 1.0
seed        = 0

# utils ---------------

def topk_frequent(x, k, largest=True):
    vals, counts = torch.unique(x, return_counts=True)   # vals ascending
    k = min(k, vals.numel())
    order = torch.argsort(counts, descending=largest, stable=True)[:k]
    return vals[order], counts[order]

def extract_negative_similarity(
    outs,
    K = 8,
    LIM = 100,
):
    minmax_records = []

    for class_idx in range(N): #. `class_idx` $\subseteq \mathbb{R}^\text{output size}$ 
        _, most_activating_idx  = torch.topk(outs[:, class_idx], k=LIM, largest=True)
        _, least_activating_idx = torch.topk(outs[:, class_idx], k=LIM, largest=False)

        most_activating_labels_open, lest_activating_labels_open = labels[most_activating_idx], labels[least_activating_idx]

        most_activating_labels, most_activating_counts = topk_frequent(most_activating_labels_open, K, largest=True)
        lest_activating_labels, lest_activating_counts = topk_frequent(lest_activating_labels_open, K, largest=True)
        
        for n, mst, lst, mc, lc in zip(range(1, N+1) , most_activating_labels, lest_activating_labels, most_activating_counts, lest_activating_counts):
            minmax_records.append({
                'target_class_idx': class_idx, 
                'order': n,
                'most_class_idx': mst.item(),
                'most_class_count': mc.item(),
                'least_class_idx': lst.item(),
                'least_class_count': lc.item(),
            })

    df = pd.DataFrame(minmax_records)
    return df


# dataset ----------
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
dataset = datasets.ImageFolder(data_dir, transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
]))

# collect all indices per class ----------
class_names = dataset.classes
if class_a not in class_names: raise ValueError(f"'{class_a}' not in classes: {class_names}")
if class_b not in class_names: raise ValueError(f"'{class_b}' not in classes: {class_names}")

label_a = class_names.index(class_a)
label_b = class_names.index(class_b)

samples = dataset.samples  # [(path, label), ...]
idx_a_all = [i for i, (_, l) in enumerate(samples) if l == label_a]
idx_b_all = [i for i, (_, l) in enumerate(samples) if l == label_b]

print(f"Class A: {class_a} (label {label_a})  -> {len(idx_a_all)} images")
print(f"Class B: {class_b} (label {label_b})  -> {len(idx_b_all)} images\n")

# pick N pairs (reproducible) ----------
rng = np.random.default_rng(pair_seed)
n_pairs = min(num_pairs, len(idx_a_all), len(idx_b_all))
pairs = [
    (int(rng.choice(idx_a_all)), int(rng.choice(idx_b_all)))
    for _ in range(n_pairs)
]

# model ----------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = model.eval().to(DEVICE)

alphas = np.linspace(0.0, 1.0, num_alphas)

# loop over pairs ----------

all_outs = []

for pair_id, (ia, ib) in enumerate(pairs):
    img_a, _ = dataset[ia]
    img_b, _ = dataset[ib]

    print("=" * 80)
    print(f"PAIR {pair_id}  (alpha = 0 -> A, alpha = 1 -> B)")
    print(f"  A: {samples[ia][0]}")
    print(f"  B: {samples[ib][0]}")

    mixed_batch = torch.stack([(1 - a) * img_a + a * img_b for a in alphas], dim=0).to(DEVICE) # [num_alphas, 3, 224, 224]

    with torch.no_grad():
        logits = model(mixed_batch) # probs.shape=(11, 1000)
