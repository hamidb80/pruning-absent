import os
import numpy as np
import torch
from torchvision import datasets, transforms, models

# =================== CONFIG ===================

CPU = 'cpu'
CUDA = 'cuda'
DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)

data_dir    = 'D:/sautkin.imagenet1kvalid/'   # <-- your ImageNet val folder
class_a     = "00104"                     # <-- first class (folder name)
class_b     = "00190"                     # <-- second class (folder name)
num_alphas  = 11
model_name  = "resnet50"                # "resnet50" | "resnet18" | "vit_b_16"
data_ratio  = 1.0
seed        = 0
# ==============================================

# ---------- dataset (your pipeline) ----------
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
dataset = datasets.ImageFolder(data_dir, transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
]))

# ---------- find one image per class ----------
class_names = dataset.classes
if class_a not in class_names: raise ValueError(f"'{class_a}' not in classes: {class_names}")
if class_b not in class_names: raise ValueError(f"'{class_b}' not in classes: {class_names}")

label_a = class_names.index(class_a)
label_b = class_names.index(class_b)

samples = dataset.samples
idx_a = next(i for i, (_, l) in enumerate(samples) if l == label_a)
idx_b = next(i for i, (_, l) in enumerate(samples) if l == label_b)

img_a, _ = dataset[idx_a]   # [3,224,224], normalized
img_b, _ = dataset[idx_b]

print(f"Class A: {class_a} (label {label_a})  <- {samples[idx_a][0]}")
print(f"Class B: {class_b} (label {label_b})  <- {samples[idx_b][0]}\n")

# ---------- model ----------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = model.eval().to(DEVICE)

# ---------- interpolate + infer ----------
alphas = np.linspace(0.0, 1.0, num_alphas)
mixed_batch = torch.stack([(1 - a) * img_a + a * img_b for a in alphas], dim=0).to(DEVICE)   
print(f"{mixed_batch.shape=}") # [num_alphas, 3, 224, 224]

with torch.no_grad():
    logits = model(mixed_batch)
    probs  = logits.softmax(dim=-1).cpu().numpy()

# ---------- report ----------
header = f"{'alpha':>6} | {'pred':>14} | {'p(A)':>8} | {'p(B)':>8} | top-3"
print(header)
print("-" * len(header))