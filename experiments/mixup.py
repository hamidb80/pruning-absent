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

# dataset ----------
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
dataset = datasets.ImageFolder(data_dir, transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
]))


# utils ---------------

def topk_frequent(x, k, largest=True):
    vals, counts = torch.unique(x, return_counts=True)   # vals ascending
    k = min(k, vals.numel())
    order = torch.argsort(counts, descending=largest, stable=True)[:k]
    return vals[order], counts[order]

# model ----------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = model.eval().to(DEVICE)



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

# loop over pairs ----------

alphas = np.linspace(0.0, 1.0, num_alphas)

for pair_id, (ia, ib) in enumerate(pairs):
    img_a, lbl_a = dataset[ia]
    img_b, lbl_b = dataset[ib]

    print("=" * 80)
    print(f"PAIR {pair_id}  (alpha = 0 -> A, alpha = 1 -> B)")
    print(f"  A: {samples[ia][0]}")
    print(f"  B: {samples[ib][0]}")

    mixed_batch = torch.stack([(1 - a) * img_a + a * img_b for a in alphas], dim=0).to(DEVICE) # [num_alphas, 3, 224, 224]

    with torch.no_grad():
        logits = model(mixed_batch) # num_alphas, rows

    print(f"{lbl_a=} {lbl_b=}")
    print(logits[:, [lbl_a, lbl_b]])