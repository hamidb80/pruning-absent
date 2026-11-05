import os

import numpy as np
import pandas as pd

import torch
from torchvision import datasets, transforms, models
from torchvision.utils import save_image  # <-- add

# CONFIG --------------------
CPU = 'cpu'
CUDA = 'cuda'
DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)

data_dir    = 'D:/sautkin.imagenet1kvalid/'
num_pairs   = 50
pair_seed   = 0
model_name  = "resnet50"
data_ratio  = 1.0
seed        = 0

out_dir     = './temp/interp_out'      # <-- add: where to save strips
save_strips = True                # <-- add

# dataset ----------
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)
dataset = datasets.ImageFolder(data_dir, transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
]))

# utils ---------------
def topk_frequent(x, k, largest=True):
    vals, counts = torch.unique(x, return_counts=True)
    k = min(k, vals.numel())
    order = torch.argsort(counts, descending=largest, stable=True)[:k]
    return vals[order], counts[order]

def unnormalize(t):
    """[N,3,H,W] normalized tensor -> [N,3,H,W] in [0,1]."""
    m = torch.tensor(mean, device=t.device).view(1, 3, 1, 1)
    s = torch.tensor(std,  device=t.device).view(1, 3, 1, 1)
    return (t * s + m).clamp(0, 1)

def hcat_strip(t):
    """[N,3,H,W] -> [3,H,N*W]  (horizontal concat)."""
    return torch.cat([t[i] for i in range(t.shape[0])], dim=2)

# model ----------
# model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
model = model.eval().to(DEVICE)

# collect all indices per class ----------
class_a     = "00104"
class_b     = "00190"
class_names = dataset.classes

if class_a not in class_names: raise ValueError(f"'{class_a}' not in classes: {class_names}")
if class_b not in class_names: raise ValueError(f"'{class_b}' not in classes: {class_names}")

label_a = class_names.index(class_a)
label_b = class_names.index(class_b)

samples = dataset.samples
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



if save_strips:
    os.makedirs(out_dir, exist_ok=True)

# loop over pairs ----------
alphas = np.linspace(0.0, 1.0, 11)

for pair_id, (ia, ib) in enumerate(pairs):
    img_a, lbl_a = dataset[ia]
    img_b, lbl_b = dataset[ib]
    print(f"{lbl_a=} {lbl_b=}")

    def focus_logits(logits):
        return logits[:, [lbl_a, lbl_b]]

    print("=" * 80)
    print(f"PAIR {pair_id}  (alpha = 0 -> A, alpha = 1 -> B)")
    print(f"  A: {samples[ia][0]}")
    print(f"  B: {samples[ib][0]}")

    mixed_batch = torch.stack([(1 - a) * img_a + a * img_b for a in alphas], dim=0).to(DEVICE) # [num_alphas, 3, 224, 224]

    with torch.no_grad():
        mixed_logits = model(mixed_batch)

    print(f"{torch.from_numpy(alphas).unsqueeze(1).shape=} {focus_logits(mixed_logits).shape=}", )
    print("mixed")
    print(alphas)
    df_mixup = pd.DataFrame(torch.cat([torch.from_numpy(alphas).unsqueeze(1), focus_logits(mixed_logits)], dim=1), columns=['alpha', lbl_a, lbl_b])
    print(df_mixup)

    # ---- horizontal strip (concat) ----
    
    # strip = hcat_strip(unnormalize(mixed_batch[[0, -1]].cpu())).unsqueeze(0)  # [3, 224, 2*224]

    # if save_strips:
    #     save_image(strip, f"{out_dir}/pair{pair_id:02d}_{class_a}_{class_b}.png")
    #     print(f"  saved strip -> {out_dir}/pair{pair_id:02d}_{class_a}_{class_b}.png")

    # with torch.no_grad():
    #     strip_logits = model(strip)
    
    # print("strip")
    # print(focus_logits(strip_logits))