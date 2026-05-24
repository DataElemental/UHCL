# UHCL Notebook Documentation

Complete guide to the `UHCL.ipynb` notebook implementation.

## Notebook Overview

This notebook implements **Unified Hierarchical Contrastive Loss (UHCL)** - a framework for learning robust representations through hierarchical data-space partitioning. The pipeline covers:

1. **Configuration & Setup** - Experiment parameters and seed management
2. **Data Loading** - Custom CIFAR-100 hierarchy classes and dataset utilities
3. **Batch Sampling** - Hierarchy-aware batch construction
4. **Model Architecture** - SmallResNet with projection head
5. **Loss Function** - UHCL implementation
6. **Training & Evaluation** - Full training loop with metrics
7. **Visualization** - t-SNE embeddings and performance analysis

---

## Section-by-Section Guide

### 1. Imports & Configuration

**Cell: Imports**
```python
import os, random, numpy as np, torch, torch.nn as nn, ...
```
- Standard dependencies for PyTorch, data processing, and visualization
- Includes scikit-learn for KNN and evaluation metrics

**Cell: CONFIG**
```python
config = {
    "experiment_name": "2222",
    "inlier_dataset": "cifar100",
    "epochs": 100,
    "batch_size": 100,
    "lr": 4e-5,
    "alpha": 0.01,          # Margin at hierarchy level 0
    "alpha_decay": 0.9,     # α[j] = α * (α_decay ** j)
    "embed_dim": 128,
    ...
}
```

**Key Parameters**:
| Parameter | Purpose | Range |
|-----------|---------|-------|
| `alpha` | Base contrastive margin | 0.01-0.5 |
| `alpha_decay` | Margin decay across hierarchy | 0.8-0.95 |
| `embed_dim` | Embedding dimensionality | 64-512 |
| `lr` | Learning rate (Adam) | 1e-5 to 1e-3 |
| `batch_size` | Samples per batch | 32-256 |
| `anchors_per_fine` | Samples per fine class in batch | 1-5 |

**Cell: set_seed() & ToTensorSafe()**
- `set_seed(39)` ensures reproducibility across runs
- `ToTensorSafe()` handles both tensor and PIL image inputs safely

---

### 2. Data Loading & Preprocessing

**Cell: IMG_SIZE & get_transforms()**

Size mapping for each dataset:
```python
IMG_SIZE = {
    "mnist": 28, "fashionmnist": 28, "emnist": 28,
    "cifar10": 32, "cifar100": 32, "svhn": 32
}
```

**Train-time augmentations** (CIFAR datasets):
- `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)`
- `RandomGrayscale(p=0.2)`
- Normalized with ImageNet mean/std: `(0.485, 0.456, 0.406)`

**Test-time** (no augmentation):
- `Resize + CenterCrop`
- ImageNet normalization

**Cell: RandomNoiseDataset**
```python
class RandomNoiseDataset(Dataset):
    def __init__(self, num: int, shape: Tuple[int, int, int]):
        self.num = num; self.shape = shape
    def __getitem__(self, idx):
        return torch.rand(self.shape), -1
```
Generates uniform random noise for the noise hierarchy level.

---

### 3. Hierarchy-Aware Data Splitting

**Cell: get_random_coarse_split()**
```python
def get_random_coarse_split(dataset, split_ratio=0.5, seed=None):
    """Randomly selects coarse labels to be inliers (e.g., 50% split)"""
    all_coarse_labels = list(dataset.coarse_label_indices.keys())
    random.shuffle(all_coarse_labels)
    n_inliers = int(len(all_coarse_labels) * split_ratio)
    return set(all_coarse_labels[:n_inliers])
```

**Cell: split_dataset_by_labels()**
```python
def split_dataset_by_labels(full_dataset, inlier_coarse_labels):
    """Splits dataset into inlier and outlier subsets based on coarse labels"""
    outlier_indices = [...]  # Collect indices of outlier coarse classes
    in_dataset = copy.copy(full_dataset)
    # Filter to keep only inlier indices
    return in_dataset, out_dataset
```

---

### 4. Custom CIFAR-100 Dataset Classes

**Cell: CIFAR100Hierarchy**
```python
class CIFAR100Hierarchy(datasets.CIFAR100):
    """Exposes both fine (100 classes) and coarse (20 superclasses) labels"""
    
    coarse_to_fine = {
        0: [4, 30, 55, 72, 95],   # aquatic mammals
        1: [1, 32, 67, 73, 91],   # fish
        ...
    }
```

**Indexing**:
```python
self.fine_label_indices    # Dict[fine_label] → List[indices]
self.coarse_label_indices  # Dict[coarse_label] → List[indices]
```

**Returns from `__getitem__`**: `(image, fine_target, coarse_target)`

**Cell: CIFAR10Indexed**
```python
class CIFAR10Indexed(datasets.CIFAR10):
    """Simpler indexing for CIFAR-10 (single label level)"""
    self.label_indices  # Dict[label] → List[indices]
```

---

### 5. Hierarchy-Aware Batch Sampling

**Cell: sample_hierarchy_batch()**

The most critical function for UHCL training:

```python
def sample_hierarchy_batch(in_dataset, out_dataset, noise_dataset,
                          batch_size, device, N=4, is_hierarchical=False):
```

**For CIFAR-100 (hierarchical=True, N=6)**:

For each fine class, creates tuple:
```
(x_0, x_1, x_2, x_3, x_4, x_5)
 ↓    ↓    ↓    ↓    ↓    ↓
anchor pos neg_fine neg_coarse outlier noise
```

**Sampling logic**:
1. **x_0 (Anchor)**: Random sample from fine class
2. **x_1 (Positive)**: Different sample, **same fine class** → same inlier
3. **x_2 (Neg Fine)**: **Different fine class, same coarse** → inlier, different type
4. **x_3 (Neg Coarse)**: **Different coarse class** → still inlier but far
5. **x_4 (Outlier)**: Sample from `out_dataset`
6. **x_5 (Noise)**: Sample from `noise_dataset`

**Label containers**:
```python
yfine_batch = [[fine0, fine0, fine2, fine3], ...]  # Only 4 entries (inlier pairs)
ycoarse_batch = [[coarse0, coarse0, coarse0, coarse3], ...]
```

**Output**:
```python
return (
    X_batch,                    # [B, N, C, H, W]
    torch.tensor(yfine_batch),  # [B, 4] - fine labels for contrastive pairs
    torch.tensor(ycoarse_batch) # [B, 4] - coarse labels
)
```

**For CIFAR-10 (hierarchical=False, N=5)**:

```
(x_0, x_1, x_2, x_3, x_4)
 ↓    ↓    ↓    ↓    ↓
anchor pos neg_other outlier noise
```

Simpler: only single label level.

---

### 6. Model Architecture

**Cell: SmallResNet**

```python
class SmallResNet(nn.Module):
    def __init__(self, embed_dim=128, pretrained=False, 
                 in_channels=3, depth=18, num_classes=None):
```

**Backbone** (shared across all samples):
- ResNet-18 or ResNet-50 (pre-trained on ImageNet optional)
- Removes final classification layer
- Output: 512 features (ResNet-18) or 2048 (ResNet-50)

**Projection Head**:
```python
nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),  # [B, C, H, W] → [B, C, 1, 1]
    nn.Flatten(),                   # [B, C]
    nn.Linear(512, 512),            # [B, 512]
    nn.ReLU(inplace=True),
    nn.Linear(512, embed_dim)       # [B, 128]
)
```

**Forward pass**:
```python
def forward(self, x, return_embeddings=True):
    z = self.backbone(x)
    z = self.projector(z)
    emb = F.normalize(z, p=2, dim=1)  # L2-normalize
    return emb, z  # (normalized, unnormalized)
```

**Optional Classification Head**:
```python
if num_classes is not None:
    self.clf_head = nn.Linear(embed_dim, num_classes)
```

Used only for post-hoc 3-class anomaly detection (Inlier/Outlier/Noise).

---

### 7. Unified Hierarchical Contrastive Loss

**Cell: uhcl()**

```python
def uhcl(emb, N=5, alpha=0.95, alpha_decay=0.8, device=None):
    """
    emb: [B, N, D] - embeddings for N hierarchy levels (L2-normalized)
    
    Enforces:
    S[i,j] > S[k,i+1] - margin[i]  for all valid i,j,k
    """
    B, total_levels, D = emb.shape
    S = torch.bmm(emb, emb.transpose(1, 2))  # [B, N, N] similarity matrix
    
    Total_loss = torch.tensor(0.0, device=device)
    
    # Define margins for each hierarchy level
    if total_levels == 6:
        margin = [15/16, 7/8, 3/4, 1/2]  # [0.9375, 0.875, 0.75, 0.5]
    else:
        margin = [7/8, 3/4, 1/2]         # [0.875, 0.75, 0.5]
    
    # For each hierarchy level j
    for j in range(1, total_levels - 1):
        alpha_j = (1 - margin[j-1]) * 0.95
        
        # For each positive pair (i,j) where i < j
        for i in range(0, j):
            # Compute hinge loss against all negatives from level i+1
            for k in range(i, j+1):
                positive_sim = S[:, i, j]
                negative_sim = S[:, k, j+1]
                hinge_loss = F.relu(negative_sim - positive_sim + alpha_j)
                Total_loss += hinge_loss.mean()
```

**Key insights**:
- Margin decreases as hierarchy gets deeper (inliers have tighter margin)
- All hierarchy levels optimized **simultaneously** (not sequentially)
- Adaptive margin: `α_j = (1 - margin[j-1]) * 0.95`

**Margin interpretation**:
- Level 0→1 (inlier pairs): margin = 0.9375 (tight, high similarity expected)
- Level 1→2 (coarse negatives): margin = 0.875
- Level 2→3 (outliers): margin = 0.75
- Level 3→4 (noise): margin = 0.5 (loose, no high similarity required)

---

### 8. Training & Evaluation

**Cell: train_clf()**

Trains a linear probe (post-hoc classifier) on frozen embeddings:

```python
def train_clf(model, in_train, out_train, noise_train, device, 
              epochs=10, batch_size=256, lr=1e-3):
    """
    Trains 3-class head: [Inlier=0, Outlier=1, Noise=2]
    using frozen backbone embeddings.
    """
    model.eval()  # Freeze backbone
    
    # Extract frozen features for all 3 classes
    in_feats, in_labels = get_feats_labels(in_train, label_val=0)
    out_feats, out_labels = get_feats_labels(out_train, label_val=1)
    noise_feats, noise_labels = get_feats_labels(noise_train, label_val=2)
    
    # Concatenate and train linear head
    X_all = torch.cat([in_feats, out_feats, noise_feats])
    y_all = torch.cat([in_labels, out_labels, noise_labels])
    
    head_opt = torch.optim.Adam(model.clf_head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for ep in range(epochs):
        for feats, labels in DataLoader(...):
            logits = model.clf_head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            head_opt.step()
```

**Cell: extract_embeddings()**

```python
def extract_embeddings(dataset, model, device, batch_size=256, hierarchical=False):
    """Extract frozen embeddings from trained model for downstream evaluation"""
    model.eval()
    
    all_features = []
    all_fine_labels = []
    all_coarse_labels = []
    
    for batch in DataLoader(dataset, ...):
        images = batch[0] if hierarchical else batch
        _, features = model(images)  # Get unnormalized features
        all_features.append(features.cpu().numpy())
    
    return np.vstack(all_features), all_fine_labels, all_coarse_labels
```

**Cell: compute_knn()**

```python
def compute_knn(train_features, train_labels, test_features, test_labels, k=20):
    """Fit KNN classifier on training features, evaluate on test"""
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(train_features, train_labels)
    
    predictions = knn.predict(test_features)
    
    metrics = {
        'accuracy': (predictions == test_labels).mean(),
        'precision': precision_score(test_labels, predictions, average='weighted'),
        'recall': recall_score(test_labels, predictions, average='weighted'),
        'f1': f1_score(test_labels, predictions, average='weighted'),
        'auroc': roc_auc_score(test_labels, predictions, average='weighted')
    }
    return predictions, metrics
```

---

### 9. Evaluation Metrics

**Evaluation Protocol** (from experiments section):

For each experiment:

1. **Inlier Accuracy**: Classification accuracy on inlier test set
   ```
   Acc = (predictions == ground_truth).mean()
   ```

2. **Outlier Detection (AUROC)**: Binary classification inlier vs outlier
   ```
   AUROC = roc_auc_score(binary_labels, anomaly_scores)
   ```

3. **Noise Rejection (AUROC)**: Binary classification meaningful vs noise
   ```
   AUROC = roc_auc_score(binary_labels, noise_scores)
   ```

4. **Macro F1-Score**: Weighted F1 across all three classes
   ```
   F1 = f1_score(labels, predictions, average='weighted')
   ```

**Anomaly scoring**:
- **Option 1 (Classifier)**: Use clf_head probability for positive class
- **Option 2 (KNN)**: Use distance to k-nearest neighbors
- **Option 3 (Distance)**: Use norm of embedding (farther = more anomalous)

---

## Complete Training Pipeline

### Step 1: Load Data
```python
# Load hierarchical dataset
train_dataset = CIFAR100Hierarchy(root='./data', train=True, download=True,
                                   transform=get_transforms('cifar100', 'train'))

# Split into inliers/outliers based on coarse classes
inlier_labels = get_random_coarse_split(train_dataset, split_ratio=0.5)
in_dataset, out_dataset = split_dataset_by_labels(train_dataset, inlier_labels)

# Create noise dataset
noise_dataset = RandomNoiseDataset(num=50000, shape=(3, 32, 32))
```

### Step 2: Initialize Model
```python
model = SmallResNet(embed_dim=128, pretrained=True, depth=18, num_classes=3)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
```

### Step 3: Training Loop
```python
for epoch in range(config['epochs']):
    for batch_idx in range(num_batches_per_epoch):
        # Sample hierarchy-aware batch
        X_batch, yfine_batch, ycoarse_batch = sample_hierarchy_batch(
            in_dataset, out_dataset, noise_dataset,
            batch_size=config['batch_size'],
            device=device,
            N=6,  # 6 levels: [anchor, pos, neg_fine, neg_coarse, outlier, noise]
            is_hierarchical=True,
            anchors_per_fine=config['anchors_per_fine']
        )
        
        # Forward pass
        # X_batch shape: [B, 6, 3, 32, 32]
        # Reshape to process all 6 hierarchy levels
        B, N, C, H, W = X_batch.shape
        X_batch_flat = X_batch.reshape(B*N, C, H, W)
        
        emb_flat, _ = model(X_batch_flat)
        emb = emb_flat.reshape(B, N, -1)  # [B, N, D]
        
        # Compute loss
        loss = uhcl(emb, N=N, alpha=config['alpha'], 
                    alpha_decay=config['alpha_decay'], device=device)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}")

### Step 4: Evaluation
```python
# Extract embeddings for evaluation
in_feats, in_labels, _, _ = extract_embeddings(
    in_test_dataset, model, device, hierarchical=True
)
out_feats, out_labels, _, _ = extract_embeddings(
    out_test_dataset, model, device, hierarchical=True
)

# Train linear probe on frozen embeddings
train_clf(model, in_dataset, out_dataset, noise_dataset, device,
          epochs=20, batch_size=256, lr=1e-3)

# Evaluate
_, metrics = compute_knn(in_feats, in_labels, 
                        out_feats, out_labels, k=20)
print(f"AUROC: {metrics['auroc']:.4f}, F1: {metrics['f1']:.4f}")
```

---

## Visualization & Analysis

### t-SNE Embedding Visualization

```python
from sklearn.manifold import TSNE

# Extract all features
all_feats, all_labels, all_coarse, _ = extract_embeddings(
    test_dataset, model, device, hierarchical=True
)

# Project to 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
feats_2d = tsne.fit_transform(all_feats)

# Plot
plt.figure(figsize=(12, 10))
scatter = plt.scatter(feats_2d[:, 0], feats_2d[:, 1], 
                     c=all_coarse, cmap='tab20', s=30, alpha=0.6)
plt.colorbar(scatter, label='Coarse Class')
plt.title('t-SNE of UHCL Embeddings (CIFAR-100)')
plt.show()
```

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

predictions, _ = compute_knn(train_feats, train_labels, 
                             test_feats, test_labels, k=20)

cm = confusion_matrix(test_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Inlier/Outlier/Noise')
plt.xlabel('Predicted')
plt.ylabel('Ground Truth')
plt.show()
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch_size`, decrease `embed_dim`, or use gradient checkpointing |
| Poor inlier accuracy | Increase `alpha_decay` (weaker hierarchy), check data split |
| Outlier not detected | Ensure outlier dataset is truly OOD, increase `alpha` margin |
| High training loss | Reduce `lr`, check data shapes, verify batch sampling |
| Embeddings collapse | Add entropy regularization or temperature scaling |

### Debugging Batch Sampling

```python
X_batch, yf, yc = sample_hierarchy_batch(...)
print(f"X_batch shape: {X_batch.shape}")  # Should be [B, 6, 3, 32, 32]
print(f"Fine labels: {yf.shape}")  # Should be [B, 4]
print(f"Coarse labels: {yc.shape}")  # Should be [B, 4]

# Verify hierarchy sampling
for i in range(min(3, len(yf))):
    print(f"Sample {i}: fine={yf[i].tolist()}, coarse={yc[i].tolist()}")
```

---

## Experimental Configurations

### Config for CIFAR-100 (N=5)
```python
config = {
    "experiment_name": "cifar100_n5",
    "inlier_dataset": "cifar100",
    "epochs": 100,
    "batch_size": 100,
    "lr": 4e-5,
    "alpha": 0.01,
    "alpha_decay": 0.9,
    "embed_dim": 128,
    "anchors_per_fine": 2,
}
```

### Config for MNIST (N=4)
```python
config = {
    "experiment_name": "mnist_n4",
    "inlier_dataset": "mnist",
    "epochs": 50,
    "batch_size": 128,
    "lr": 1e-4,
    "alpha": 0.05,
    "alpha_decay": 0.85,
    "embed_dim": 64,
    "anchors_per_fine": 3,
}
```

### Config for CIFAR-10/SVHN (N=4)
```python
config = {
    "experiment_name": "cifar10_svhn_n4",
    "inlier_dataset": "cifar10",
    "epochs": 80,
    "batch_size": 100,
    "lr": 2e-4,
    "alpha": 0.02,
    "alpha_decay": 0.88,
    "embed_dim": 128,
    "anchors_per_fine": 2,
}
```

---

## References

Key papers referenced in the notebook:

1. **SimCLR**: Chen et al., "A Simple Framework for Contrastive Learning"
2. **Hierarchical Classification**: Zhang et al., "Hierarchical Multi-label Classification"
3. **Anomaly Detection**: Ghafoori et al., "Deep Anomaly Detection"
4. **OOD Detection**: Liang et al., "ODIN: Out-of-Distribution Indicator"

See `references.bib` for complete bibliography.

---

## Tips for Best Results

1. **Data quality**: Ensure outlier dataset is truly out-of-distribution
2. **Hierarchy design**: Hierarchy should reflect meaningful semantic levels
3. **Hyperparameter tuning**: Start with provided configs, adjust `alpha` based on validation
4. **Batch sampling**: Use `anchors_per_fine ≥ 2` for stable gradients
5. **Evaluation**: Always evaluate on separate test set, not training set

---

Last updated: 2025-05-24
Author: Hosein Moradi
