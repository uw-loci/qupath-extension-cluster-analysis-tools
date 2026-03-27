"""
[TEST FEATURE] Semi-supervised VAE for cell classification.

Trains a variational autoencoder with an optional classifier head on
cell marker measurements. When some cells have user-assigned labels,
the model learns a latent space that both reconstructs measurements
AND separates labeled cell types (semi-supervised).

Architecture follows the scANVI pattern (Xu et al. 2021, Molecular Systems Biology)
adapted for continuous protein measurements (Gaussian likelihood instead of ZINB).

Training features adapted from the DL pixel classifier extension:
- Class weighting (inverse-frequency) for imbalanced cell populations
- Validation split (20%) with stratified sampling
- Early stopping with best-model restoration
- OneCycleLR learning rate scheduling
- Mixed precision training (AMP) on CUDA
- Gradient clipping (max_norm=1.0)
- AdamW optimizer with weight decay

Inputs (injected by Appose 0.10.0):
  measurements: NDArray (N_cells x N_markers, float64)  [measurement mode]
  tile_images: NDArray (N_cells x C x H x W, float32)   [tile mode]
  marker_names: list[str]
  labels: list[int] -- class index per cell (-1 = unlabeled)
  label_names: list[str] -- class name for each index
  input_mode: str ("measurements" or "tiles")
  latent_dim: int (default 16)
  n_epochs: int (default 100)
  learning_rate: float (default 0.001)
  batch_size: int (default 128)
  supervision_weight: float (default 1.0)
  normalization: str ("zscore", "minmax", "none")
  validation_split: float (default 0.2)
  early_stopping_patience: int (default 15, 0 = disabled)
  enable_class_weights: bool (default True)
  enable_augmentation: bool (default True)

Optional inputs:
  model_save_path: str -- path to save trained model checkpoint
  n_channels: int -- number of image channels (tile mode)
  tile_size: int -- tile size in pixels (tile mode)

Outputs (via task.outputs):
  latent_features: NDArray (N_cells x latent_dim, float32)
  predicted_labels: NDArray (N_cells,) int32
  prediction_confidence: NDArray (N_cells,) float32
  n_classes: int
  final_recon_loss: float
  final_class_accuracy: float
  best_val_accuracy: float
  best_epoch: int
  model_state_base64: str -- base64-encoded state dict
"""
import sys
import json
import logging
import base64
import io

logger = logging.getLogger("qpcat.autoencoder")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.nn.utils import clip_grad_norm_
from appose import NDArray as PyNDArray

# ==================== VAE Architecture ====================

class CellVAE(nn.Module):
    """
    Variational Autoencoder with optional semi-supervised classifier head.

    Encoder: input -> 128 -> 64 -> (mu, logvar) of size latent_dim
    Decoder: latent_dim -> 64 -> 128 -> input (Gaussian reconstruction)
    Classifier: latent_dim -> n_classes (optional, for labeled cells)
    """

    def __init__(self, n_markers, latent_dim, n_classes=0):
        super().__init__()
        self.n_markers = n_markers
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        # Encoder
        self.enc1 = nn.Linear(n_markers, 128)
        self.enc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.dec1 = nn.Linear(latent_dim, 64)
        self.dec2 = nn.Linear(64, 128)
        self.dec_out = nn.Linear(128, n_markers)

        # Classifier head (only if supervised/semi-supervised)
        if n_classes > 0:
            self.classifier = nn.Linear(latent_dim, n_classes)
        else:
            self.classifier = None

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        return self.dec_out(h)

    def classify(self, z):
        if self.classifier is None:
            return None
        return self.classifier(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        class_logits = self.classify(z)
        return recon, mu, logvar, z, class_logits


class ConvCellVAE(nn.Module):
    """
    Convolutional VAE for tile-based cell classification.

    Takes multi-channel image tiles (NCHW format) and learns a latent
    representation via convolutional encoder/decoder. Supports variable
    channel counts for multiplexed imaging (IMC, CODEX, IF panels).

    Encoder: Conv2d layers -> AdaptiveAvgPool -> flatten -> (mu, logvar)
    Decoder: Linear -> unflatten -> ConvTranspose2d layers -> output
    Classifier: latent_dim -> n_classes (optional)
    """

    def __init__(self, n_channels, tile_size, latent_dim, n_classes=0):
        super().__init__()
        self.n_channels = n_channels
        self.tile_size = tile_size
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        # Encoder: conv layers that reduce spatial dims
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> (B, 128, 1, 1)
            nn.Flatten(),             # -> (B, 128)
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder: reconstruct spatial output
        self.dec_spatial = max(tile_size // 8, 1)
        self.dec_fc = nn.Linear(latent_dim, 128 * self.dec_spatial * self.dec_spatial)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, n_channels, 3, stride=2, padding=1, output_padding=1),
        )

        # Classifier head
        if n_classes > 0:
            self.classifier = nn.Linear(latent_dim, n_classes)
        else:
            self.classifier = None

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec_fc(z))
        h = h.view(-1, 128, self.dec_spatial, self.dec_spatial)
        h = self.decoder(h)
        return h[:, :, :self.tile_size, :self.tile_size]

    def classify(self, z):
        if self.classifier is None:
            return None
        return self.classifier(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        class_logits = self.classify(z)
        return recon, mu, logvar, z, class_logits


def vae_loss(recon, x, mu, logvar):
    """Gaussian reconstruction loss + KL divergence."""
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl_loss


# ==================== Early Stopping ====================

class EarlyStopping:
    """
    Stops training when validation metric stops improving.
    Adapted from the DL pixel classifier extension.
    """

    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_state = None
        self.should_stop = False

    def step(self, score, epoch, model):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            logger.info("Restored best model from epoch %d (score=%.4f)",
                        self.best_epoch + 1, self.best_score)


# ==================== Data Augmentation ====================

def augment_measurements(data, noise_std=0.05, scale_range=0.1):
    """Simple measurement-space augmentation: Gaussian noise + random scaling."""
    noise = torch.randn_like(data) * noise_std
    scale = 1.0 + (torch.rand(data.shape[1], device=data.device) - 0.5) * 2 * scale_range
    return data * scale + noise


# ==================== Main Script ====================

# 1. Parse inputs
try:
    mode = input_mode
except NameError:
    mode = "measurements"

use_tiles = (mode == "tiles")

if use_tiles:
    raw_tiles = tile_images.ndarray().copy().astype(np.float32)
    n_cells = raw_tiles.shape[0]
    img_channels = int(n_channels)
    img_tile_size = int(tile_size)
    logger.info("Tile mode: %d cells, %d channels, %dx%d tiles",
                n_cells, img_channels, img_tile_size, img_tile_size)
    data = None
    n_markers = 0
else:
    data = measurements.ndarray().copy().astype(np.float32)
    n_cells, n_markers = data.shape
    raw_tiles = None
    img_channels = 0
    img_tile_size = 0

label_array = np.array(labels, dtype=np.int32)
class_names = list(label_names)
n_classes = len(class_names)

logger.info("Received %d cells, %d classes", n_cells, n_classes)

# Parse optional parameters with defaults
try:
    ldim = latent_dim
except NameError:
    ldim = 16
try:
    epochs = n_epochs
except NameError:
    epochs = 100
try:
    lr = learning_rate
except NameError:
    lr = 0.001
try:
    bs = batch_size
except NameError:
    bs = 128
try:
    sup_weight = supervision_weight
except NameError:
    sup_weight = 1.0
try:
    norm_method = normalization
except NameError:
    norm_method = "zscore"
try:
    save_path = model_save_path
except NameError:
    save_path = None
try:
    val_split = validation_split
except NameError:
    val_split = 0.2
try:
    es_patience = early_stopping_patience
except NameError:
    es_patience = 15
try:
    do_class_weights = enable_class_weights
except NameError:
    do_class_weights = True
try:
    do_augmentation = enable_augmentation
except NameError:
    do_augmentation = True

# Count labeled vs unlabeled
labeled_mask = label_array >= 0
n_labeled = int(labeled_mask.sum())
n_unlabeled = n_cells - n_labeled
has_labels = n_labeled > 0

logger.info("Labeled: %d, Unlabeled: %d, Classes: %d", n_labeled, n_unlabeled, n_classes)
if has_labels:
    for i, name in enumerate(class_names):
        count = int((label_array == i).sum())
        logger.info("  Class %d (%s): %d cells", i, name, count)

# 2. Normalize
task.update("Normalizing...", current=0, maximum=epochs + 2)

mean_stat = None
std_stat = None
dmin_stat = None
dmax_stat = None

if use_tiles:
    if norm_method == "zscore":
        mean_stat = raw_tiles.mean(axis=(0, 2, 3), keepdims=True)
        std_stat = raw_tiles.std(axis=(0, 2, 3), keepdims=True)
        std_stat[std_stat == 0] = 1
        data_norm_tiles = (raw_tiles - mean_stat) / std_stat
        mean_stat = mean_stat.squeeze()
        std_stat = std_stat.squeeze()
    elif norm_method == "minmax":
        dmin_stat = raw_tiles.min(axis=(0, 2, 3), keepdims=True)
        dmax_stat = raw_tiles.max(axis=(0, 2, 3), keepdims=True)
        drange = dmax_stat - dmin_stat
        drange[drange == 0] = 1
        data_norm_tiles = (raw_tiles - dmin_stat) / drange
        dmin_stat = dmin_stat.squeeze()
        dmax_stat = dmax_stat.squeeze()
    else:
        data_norm_tiles = raw_tiles.copy()
    data_norm = None
else:
    if norm_method == "zscore":
        mean_stat = data.mean(axis=0)
        std_stat = data.std(axis=0)
        std_stat[std_stat == 0] = 1
        data_norm = (data - mean_stat) / std_stat
    elif norm_method == "minmax":
        dmin_stat = data.min(axis=0)
        dmax_stat = data.max(axis=0)
        drange = dmax_stat - dmin_stat
        drange[drange == 0] = 1
        data_norm = (data - dmin_stat) / drange
    else:
        data_norm = data.copy()
    data_norm_tiles = None

# 3. Compute class weights (inverse-frequency, adapted from DL pixel classifier)
class_weight_tensor = None
if has_labels and do_class_weights and n_classes > 1:
    class_counts = np.array([int((label_array == i).sum()) for i in range(n_classes)])
    # Avoid division by zero for missing classes
    class_counts = np.maximum(class_counts, 1)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * n_classes  # normalize so mean weight = 1
    class_weight_tensor = torch.tensor(weights, dtype=torch.float32)
    logger.info("Class weights: %s", {name: "%.2f" % w for name, w in zip(class_names, weights)})

# 4. Train/validation split (stratified by label where possible)
from model_utils import detect_device
device = detect_device()

if use_tiles:
    data_tensor = torch.tensor(data_norm_tiles, dtype=torch.float32)
else:
    data_tensor = torch.tensor(data_norm, dtype=torch.float32)

label_tensor = torch.tensor(label_array, dtype=torch.long)

# Stratified split: maintain class proportions in train and val
if val_split > 0 and n_cells > 20:
    from sklearn.model_selection import train_test_split
    indices = np.arange(n_cells)
    # Use labeled cells for stratification, unlabeled as a single group
    strat_labels = label_array.copy()
    strat_labels[strat_labels < 0] = n_classes  # group all unlabeled together
    try:
        train_idx, val_idx = train_test_split(
            indices, test_size=val_split, random_state=42, stratify=strat_labels)
    except ValueError:
        # Stratification fails if any class has <2 samples
        train_idx, val_idx = train_test_split(
            indices, test_size=val_split, random_state=42)
    train_dataset = Subset(TensorDataset(data_tensor, label_tensor), train_idx)
    val_dataset = Subset(TensorDataset(data_tensor, label_tensor), val_idx)
    n_val = len(val_idx)
    logger.info("Train/val split: %d/%d (%.0f%% validation)", len(train_idx), n_val, val_split * 100)
else:
    train_dataset = TensorDataset(data_tensor, label_tensor)
    val_dataset = None
    n_val = 0
    train_idx = np.arange(n_cells)
    val_idx = np.array([], dtype=int)

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False) if val_dataset else None

# 5. Build model and optimizer
if use_tiles:
    model = ConvCellVAE(img_channels, img_tile_size, ldim,
                        n_classes if has_labels else 0).to(device)
else:
    model = CellVAE(n_markers, ldim, n_classes if has_labels else 0).to(device)

# AdamW with weight decay (adapted from DL pixel classifier)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

# OneCycleLR scheduler (adapted from DL pixel classifier)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr, epochs=epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.3, anneal_strategy='cos')

# Mixed precision (CUDA only)
use_amp = (device == "cuda")
scaler = torch.amp.GradScaler("cuda") if use_amp else None

# Early stopping
early_stopper = EarlyStopping(patience=es_patience) if es_patience > 0 else None

if class_weight_tensor is not None:
    class_weight_tensor = class_weight_tensor.to(device)

logger.info("Training config: latent=%d, epochs=%d, lr=%g, bs=%d, val=%.0f%%, "
            "early_stop=%s, class_weights=%s, augment=%s, AMP=%s",
            ldim, epochs, lr, bs, val_split * 100,
            "patience=%d" % es_patience if es_patience > 0 else "off",
            "on" if class_weight_tensor is not None else "off",
            "on" if do_augmentation else "off",
            "on" if use_amp else "off")

# 6. Training loop
best_val_acc = -1.0
best_epoch = 0

for epoch in range(epochs):
    # --- Train phase ---
    model.train()
    total_recon = 0.0
    total_kl = 0.0
    train_correct = 0
    train_labeled = 0
    n_batches = 0

    for batch_data, batch_labels in train_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        # Data augmentation (measurement mode only, applied on GPU)
        if do_augmentation and not use_tiles:
            batch_data = augment_measurements(batch_data)

        with torch.amp.autocast("cuda", enabled=use_amp):
            recon, mu, logvar, z, class_logits = model(batch_data)
            recon_loss, kl_loss = vae_loss(recon, batch_data, mu, logvar)
            loss = recon_loss + kl_loss

            # Classification loss with class weights
            if has_labels and class_logits is not None:
                labeled_in_batch = batch_labels >= 0
                if labeled_in_batch.any():
                    class_loss = F.cross_entropy(
                        class_logits[labeled_in_batch],
                        batch_labels[labeled_in_batch],
                        weight=class_weight_tensor)
                    loss = loss + sup_weight * class_loss

                    preds = class_logits[labeled_in_batch].argmax(dim=1)
                    train_correct += (preds == batch_labels[labeled_in_batch]).sum().item()
                    train_labeled += labeled_in_batch.sum().item()

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_batches += 1

    avg_recon = total_recon / max(n_batches, 1)
    avg_kl = total_kl / max(n_batches, 1)
    train_acc = train_correct / max(train_labeled, 1)

    # --- Validation phase ---
    val_acc = -1.0
    if val_loader and has_labels:
        model.eval()
        val_correct = 0
        val_labeled = 0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                _, mu_v, _, _, class_logits_v = model(batch_data)
                if class_logits_v is not None:
                    labeled_v = batch_labels >= 0
                    if labeled_v.any():
                        preds_v = class_logits_v[labeled_v].argmax(dim=1)
                        val_correct += (preds_v == batch_labels[labeled_v]).sum().item()
                        val_labeled += labeled_v.sum().item()
        val_acc = val_correct / max(val_labeled, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

    # Early stopping check (on val accuracy if available, else train loss)
    if early_stopper is not None:
        if val_acc >= 0:
            early_stopper.step(val_acc, epoch, model)
        else:
            early_stopper.step(-avg_recon, epoch, model)  # lower recon = better

    # Logging
    if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
        msg = "Epoch %d/%d: recon=%.4f, KL=%.4f" % (epoch + 1, epochs, avg_recon, avg_kl)
        if has_labels:
            msg += ", train_acc=%.1f%%" % (train_acc * 100)
        if val_acc >= 0:
            msg += ", val_acc=%.1f%%" % (val_acc * 100)
        current_lr = optimizer.param_groups[0]['lr']
        msg += ", lr=%.2e" % current_lr
        logger.info(msg)
        task.update(msg, current=epoch + 1, maximum=epochs + 2)

    if early_stopper and early_stopper.should_stop:
        logger.info("Early stopping at epoch %d (patience=%d, best=%d)",
                    epoch + 1, es_patience, early_stopper.best_epoch + 1)
        task.update("Early stopping at epoch %d (best: epoch %d)" %
                    (epoch + 1, early_stopper.best_epoch + 1))
        break

# Restore best model if early stopping was used
if early_stopper and early_stopper.best_state is not None:
    early_stopper.restore_best(model)
    best_epoch = early_stopper.best_epoch

# Use final training accuracy for reporting
accuracy = train_acc

# 7. Encode ALL cells (both train and val) and predict
task.update("Encoding all cells...", current=epochs + 1, maximum=epochs + 2)
model.eval()

with torch.no_grad():
    all_data = data_tensor.to(device)
    mu_all, _ = model.encode(all_data)
    latent = mu_all.cpu().numpy()

    pred_labels = np.full(n_cells, -1, dtype=np.int32)
    pred_conf = np.zeros(n_cells, dtype=np.float32)

    if has_labels and model.classifier is not None:
        logits = model.classify(mu_all)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        pred_labels = probs.argmax(axis=1).astype(np.int32)
        pred_conf = probs.max(axis=1).astype(np.float32)

logger.info("Encoding complete: %d cells x %d latent dims", n_cells, ldim)

# 8. Save model checkpoint
model_b64 = ""
if save_path:
    torch.save(model.state_dict(), save_path)
    logger.info("Model saved to %s", save_path)

buf = io.BytesIO()
checkpoint_data = {
    'state_dict': model.state_dict(),
    'input_mode': mode,
    'latent_dim': ldim,
    'n_classes': n_classes,
    'class_names': class_names,
    'normalization': norm_method,
    'norm_params': {
        'mean': mean_stat.tolist() if mean_stat is not None else None,
        'std': std_stat.tolist() if std_stat is not None else None,
        'min': dmin_stat.tolist() if dmin_stat is not None else None,
        'max': dmax_stat.tolist() if dmax_stat is not None else None,
    },
}
if use_tiles:
    checkpoint_data['n_channels'] = img_channels
    checkpoint_data['tile_size'] = img_tile_size
else:
    checkpoint_data['n_markers'] = n_markers
    checkpoint_data['marker_names'] = list(marker_names)

torch.save(checkpoint_data, buf)
model_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

# 9. Package outputs
task.update("Packaging results...", current=epochs + 2, maximum=epochs + 2)

latent_nd = PyNDArray(dtype="float32", shape=[n_cells, ldim])
np.copyto(latent_nd.ndarray(), latent.astype(np.float32))
task.outputs["latent_features"] = latent_nd

pred_nd = PyNDArray(dtype="int32", shape=[n_cells])
np.copyto(pred_nd.ndarray(), pred_labels)
task.outputs["predicted_labels"] = pred_nd

conf_nd = PyNDArray(dtype="float32", shape=[n_cells])
np.copyto(conf_nd.ndarray(), pred_conf)
task.outputs["prediction_confidence"] = conf_nd

task.outputs["n_classes"] = n_classes
task.outputs["final_recon_loss"] = float(avg_recon)
task.outputs["final_class_accuracy"] = float(accuracy) if has_labels else -1.0
task.outputs["best_val_accuracy"] = float(best_val_acc) if best_val_acc >= 0 else -1.0
task.outputs["best_epoch"] = int(best_epoch + 1)
task.outputs["model_state_base64"] = model_b64
task.outputs["label_names_json"] = json.dumps(class_names)

logger.info("[TEST FEATURE] Autoencoder training complete (best epoch: %d)", best_epoch + 1)
