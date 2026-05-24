# UHCL Notebook Documentation

This document provides detailed explanations of key components in `UHCL.ipynb`.

## Section 1: Imports & Configuration

### Configuration Dictionary
```python
config = {
    "experiment_name": "2222",
    "inlier_dataset": "cifar100",
    "outlier_dataset": "cifar100",
    "epochs": 100,
    "batch_size": 100,
    "anchors_per_fine": 2,
    "lr": 4e-5,
    "alpha": 0.01,              # Margin scale in contrastive loss
    "alpha_decay": 0.9,         # Decays margin across hierarchy levels
    "embed_dim": 128,           # Final embedding dimension (L2-normalized)
    "num_workers": 8,
    "pretrained": True,         # Use ImageNet pre-trained backbone
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "ckpt_dir": "./content/checkpoints",
    "resume_ckpt_path": None,   # For resuming from checkpoint
}
```

### Key Hyperparameters

| Parameter | Role | Typical Value |
|-----------|------|---------------|
| `alpha` | Margin between positive and negative similarities | 0.8 - 0.95 |
| `alpha_decay` | Factor for margin decay per hierarchy level | 0.8 - 0.9 |
| `embed_dim` | Dimension of L2-normalized embeddings | 128 |
| `lr` | Learning rate for Adam optimizer | 4e-5 |
| `anchors_per_fine` | Number of anchors sampled per fine class | 2 |

---

## Section 2: Data Loaders & Transforms

### Transform Strategy

The notebook implements dataset-specific augmentation:

```
Grayscale Datasets (MNIST, FashionMNIST, EMNIST):
    - Minimal augmentation: Just normalization
    - Reasoning: Limited color information to augment

Color Image Datasets (CIFAR-10, CIFAR-100):
    - ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    - RandomGrayscale(p=0.2)
    - ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
```

### Custom Dataset Classes

#### `CIFAR100Hierarchy`
Extends torchvision's CIFAR100 to expose both fine (100 classes) and coarse (20 superclasses) labels.

**Key Features**:
- Pre-computed `fine_label_indices` & `coarse_label_indices` for efficient sampling
- `__getitem__()` returns `(image, fine_label, coarse_label)`
- Enables hierarchical batch construction

**Usage**:
```python
dataset = CIFAR100Hierarchy(root='./data', train=True, transform=transform)
# Coarse hierarchy: 0=aquatic mammals, 1=fish, ..., 19=vehicles 2
# Each coarse class has 5 fine classes (5×20=100 total)
```

#### `CIFAR10Indexed`
Flat wrapper around CIFAR10 with pre-computed label indices for faster sampling.

#### `RandomNoiseDataset`
Generates random uniform noise in `[0, 1]` with specified shape.

```python
noise_dataset = RandomNoiseDataset(num=50000, shape=(3, 32, 32))
```

---

## Section 3: Hierarchical Batch Sampling

### Purpose
Create training tuples that respect the data-space hierarchy: `Noise → Outliers → Inliers → Positive Inliers`.

### Function: `sample_hierarchy_batch()`

**Inputs**:
- `in_dataset`: Inlier dataset (with fine/coarse indices)
- `out_dataset`: Outlier dataset
- `noise_dataset`: Noise dataset
- `batch_size`: Batch size
- `device`: PyTorch device
- `N`: Number of hierarchy levels
- `is_hierarchical`: If True, uses coarse labels

**For Hierarchical Datasets (CIFAR-100, is_hierarchical=True)**:

Each tuple contains:
```python
(x0, x1, x2, x3, x4, x5)  # Shape: [6, 3, 32, 32]

x0: Anchor (any fine class)
x1: Positive (same fine class as x0)
x2: Negative inlier (same coarse as x0, different fine)
x3: Negative inlier (different coarse from x0)
x4: Outlier (out-of-distribution)
x5: Noise (random pattern)
```

**For Flat Datasets (CIFAR-10, is_hierarchical=False)**:

```python
(x0, x1, x2, x3, x4)  # Shape: [5, 3, 32, 32]

x0: Anchor
x1: Positive (same class)
x2: Negative inlier (different class)
x3: Outlier
x4: Noise
```

**Sampling Strategy**:
```python
# For each fine class in in_dataset:
for fine_label in fine_labels:
    fine_indices = in_dataset.fine_label_indices[fine_label]
    
    # Sample anchors
    anchors = np.random.choice(fine_indices, anchors_per_fine)
    
    for anchor_idx in anchors:
        x0 = in_dataset[anchor_idx]
        
        # x1: Same fine, different instance
        pos_candidates = [i for i in fine_indices if i != anchor_idx]
        x1 = in_dataset[random.choice(pos_candidates)]
        
        # x2: Same coarse, different fine
        coarse_candidates = [i for i in coarse_indices[coarse0] if targets[i] != fine0]
        x2 = in_dataset[random.choice(coarse_candidates)]
        
        # x3: Different coarse
        x3 = in_dataset[random.choice(diff_coarse_candidates)]
        
        # x4: Outlier
        x4 = out_dataset[random.randint(len(out_dataset))]
        
        # x5: Noise
        x5 = noise_dataset[random.randint(len(noise_dataset))]
        
        X_batch.append(torch.stack([x0, x1, x2, x3, x4, x5]))
```

---

## Section 4: Model Architecture

### `SmallResNet` Class

**Constructor Parameters**:
- `embed_dim`: Embedding dimension (default: 128)
- `pretrained`: Use ImageNet weights (default: False)
- `in_channels`: Input channels (1 for grayscale, 3 for RGB)
- `depth`: ResNet depth (18 or 50)
- `num_classes`: If specified, adds classification head

**Architecture**:

```
Input [B, C, H, W]
    ↓
ResNet18/50 Backbone (removes final classification layer)
    ↓
AdaptiveAvgPool2d(1, 1) → [B, 512 (or 2048), 1, 1]
    ↓
Flatten() → [B, 512 (or 2048)]
    ↓
Linear(512, 512) → ReLU → Linear(512, embed_dim)
    ↓
L2 Normalization → [B, embed_dim] (unit vectors on hypersphere)
    ↓
[Optional] Linear(embed_dim, num_classes) → Logits [B, num_classes]
```

**Key Design Choices**:

1. **AdaptiveAvgPool2d**: Ensures 1×1 spatial dims regardless of input resolution
2. **L2 Normalization**: Maps embeddings to unit hypersphere (important for contrastive learning)
3. **Two-layer MLP**: Projects backbone features through 512-dim intermediate

**Example Usage**:

```python
# ResNet18 with 128-dim embeddings, pretrained on ImageNet
model = SmallResNet(embed_dim=128, pretrained=True, depth=18, num_classes=3)
model = model.to(device)

# Forward pass
emb, z = model(x)  # emb: [B, 128] (normalized)
                   # z: [B, 128] (before normalization)
```

---

## Section 5: UHCL Loss Function

### Loss Formulation

Given normalized embeddings for a batch tuple: `[B, N, D]` where:
- B = batch size
- N = hierarchy levels (e.g., 6 for the 6-tuple)
- D = embedding dimension

**Step 1: Compute Similarity Matrix**
```python
S = torch.bmm(emb, emb.transpose(1, 2))  # [B, N, N]
S[b, i, j] = emb[b, i] · emb[b, j]
```

**Step 2: Enforce Margin Constraints**

For each hierarchy level j and each preceding level i:
```
max(0, S[b, k, j+1] - S[b, i, j] + α) 
∀ k ∈ [i, j]
```

This means:
- Positive similarity at (i, j) should be ≥ negative similarity at (k, j+1) + margin α
- The margin α ensures a "gap" between positives and negatives

**Step 3: Normalize and Aggregate**

```python
total_loss = 0
for j in range(1, N-1):  # Each level as "positive level"
    for i in range(0, j):  # Each preceding level
        for k in range(i, j+1):  # All negatives
            hinge_loss = max(0, S[k, j+1] - S[i, j] + α)
            total_loss += hinge_loss / (j - i + 1)  # Normalize by # negatives
        total_loss *= (1 / j)  # Normalize by # positive levels
```

### Code Implementation

```python
def uhcl(emb, N=5, alpha=0.95, alpha_decay=0.8, device=None):
    """
    emb: [B, N, D] normalized embeddings
    N: number of hierarchy levels
    alpha: margin scale
    alpha_decay: margin decay factor per level
    """
    B, total_levels, D = emb.shape
    S = torch.bmm(emb, emb.transpose(1, 2))  # [B, N, N]
    
    Total_loss = torch.tensor(0.0, device=device)
    
    # Margin loss: anchor vs. final level
    total_margin_loss = F.relu(S[:, 0, total_levels-1]).mean()
    
    if total_levels == 6:
        margin = [15/16, 7/8, 3/4, 1/2]  # Adaptive margins
    else:
        margin = [7/8, 3/4, 1/2]
    
    # Hierarchical losses
    for j in range(1, total_levels - 1):
        alpha_j = (1 - margin[j-1]) * 0.95  # Dynamic margin
        hierarchy_j_loss = torch.tensor(0.0, device=device)
        
        for i in range(0, j):
            partial_loss = torch.tensor(0.0, device=device)
            
            for k in range(i, j + 1):
                positive_sim = S[:, i, j]
                negative_sim = S[:, k, j + 1]
                
                # Hinge loss
                hinge_loss = F.relu(negative_sim - positive_sim + alpha_j)
                partial_loss += hinge_loss.mean()
            
            lambda_ij = 1 / (j - i + 1)
            hierarchy_j_loss += partial_loss * lambda_ij
        
        lambda_j = 1 / j
        Total_loss += hierarchy_j_loss * lambda_j
    
    return Total_loss + total_margin_loss
```

### Key Properties

1. **Batch-wise Computation**: Each batch sample has independent similarity matrices
2. **Margin Adaptation**: α_j changes based on hierarchy level
3. **Normalization**: Coefficients λ_ij and λ_j normalize by #pairs
4. **Extensibility**: Works with any N levels

---

## Section 6: Evaluation

### `train_clf()`: Linear Probe Training

**Purpose**: Train a lightweight classification head on frozen backbone features.

**Process**:
1. Set backbone to eval mode (no gradient updates)
2. Extract features from all training data
3. Train a linear classifier on these frozen features
4. Evaluate on validation data

```python
# Feature extraction
in_feats, in_labels = get_feats_labels(in_train, label=0)
out_feats, out_labels = get_feats_labels(out_train, label=1)
noise_feats, noise_labels = get_feats_labels(noise_train, label=2)

# Concatenate all features
X_all = torch.cat([in_feats, out_feats, noise_feats])
y_all = torch.cat([in_labels, out_labels, noise_labels])

# Train linear layer
for epoch in range(epochs):
    logits = model.clf_head(features)  # [B, 3]
    loss = CrossEntropyLoss()(logits, labels)
    loss.backward()  # Only updates clf_head
    optimizer.step()
```

### `extract_embeddings()`: Feature Extraction

Extracts learned representations for downstream evaluation.

```python
def extract_embeddings(dataset, model, device, batch_size=256, hierarchical=False):
    """
    Returns:
        all_features: [N, D] numpy array of embeddings
        all_fine_labels: [N] class labels
        all_coarse_labels: [N] superclass labels
        all_classifier_probs: [N, num_classes] classification probabilities
    """
    model.eval()
    all_features = []
    
    for batch in dataloader:
        images = batch[0].to(device)
        _, features_t = model(images)  # Get un-normalized features
        all_features.append(features_t.cpu().numpy())
    
    return np.vstack(all_features), labels, coarse_labels, probs
```

### `compute_knn()`: K-NN Evaluation

Trains a k-NN classifier on learned embeddings.

```python
def compute_knn(train_features, train_labels, test_features, test_labels, k=20):
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(train_features, train_labels)
    predictions = knn.predict(test_features)
    
    # Compute metrics
    accuracy = (predictions == test_labels).mean()
    precision = precision_score(test_labels, predictions, average='weighted')
    recall = recall_score(test_labels, predictions, average='weighted')
    f1 = f1_score(test_labels, predictions, average='weighted')
    
    return accuracy, precision, recall, f1
```

---

## Section 7: Training Loop (Typical Usage)

```python
# 1. Setup
model = SmallResNet(embed_dim=128, pretrained=True, num_classes=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=4e-5)

# 2. Training
for epoch in range(config['epochs']):
    # Sample hierarchical batch
    X_batch, yfine, ycoarse = sample_hierarchy_batch(
        in_dataset, out_dataset, noise_dataset,
        batch_size=config['batch_size'],
        device=device,
        N=4,
        is_hierarchical=True
    )
    
    # Forward pass: X_batch shape [B, 6, 3, 32, 32]
    B = X_batch.shape[0]
    X_flat = X_batch.view(B * 6, 3, 32, 32)  # Flatten batch×hierarchy
    emb_flat, z_flat = model(X_flat)
    emb = emb_flat.view(B, 6, -1)  # Reshape back to [B, 6, 128]
    
    # Compute loss
    loss = uhcl(emb, N=6, alpha=config['alpha'], device=device)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Periodic evaluation
    if (epoch + 1) % 10 == 0:
        features, _, _, _ = extract_embeddings(test_dataset, model, device)
        # Evaluate metrics
```

---

## Common Issues & Solutions

### Issue 1: "RuntimeError: expected 3D input (got 2D input)"
**Cause**: Embeddings not reshaped to [B, N, D] before UHCL loss.

**Solution**:
```python
# WRONG:
emb_flat = model(X_batch)  # [B*6, 128]
loss = uhcl(emb_flat, ...)

# CORRECT:
emb_flat = model(X_batch)  # [B*6, 128]
emb = emb_flat.view(B, 6, 128)  # [B, 6, 128]
loss = uhcl(emb, ...)
```

### Issue 2: Slow batch sampling
**Cause**: `sample_hierarchy_batch()` loops over all classes.

**Solution**: Use pre-computed indices and vectorized operations.

### Issue 3: Poor outlier detection on unseen OOD
**Cause**: Outlier diversity during training is limited.

**Solution**: Use multiple diverse outlier sources or harder negative mining.

---

## Reference: Key Equations

### Similarity Matrix
$$S_{ij}^{(b)} = \mathbf{z}_i^{(b)} \cdot \mathbf{z}_j^{(b)}$$

### UHCL Loss
$$\mathcal{L} = \sum_{j=1}^{N-1} \sum_{i=0}^{j-1} \lambda_{ij} \sum_{k=i}^{j} \max(S_{kj+1} - S_{ij} + \alpha_j, 0)$$

### Normalization Coefficients
$$\lambda_{ij} = \frac{1}{j - i + 1}, \quad \lambda_j = \frac{1}{j}$$

---

## Further Reading

- **Contrastive Learning**: Khosla et al., "Supervised Contrastive Learning" (ICML 2020)
- **Anomaly Detection**: Wang et al., "HSCL: Hierarchical Semi-supervised Contrastive Learning"
- **Hierarchical Learning**: Zhang et al., "Hierarchical Multi-label Classification via InfoNCE"

---

**Document Version**: 1.0  
**Last Updated**: May 2026
