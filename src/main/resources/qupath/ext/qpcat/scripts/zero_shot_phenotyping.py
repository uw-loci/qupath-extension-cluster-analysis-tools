"""
Zero-shot cell/tile phenotyping using vision-language models.

Uses BiomedCLIP (MIT License, Microsoft) for text-image similarity scoring.
Each cell's tile image is compared against user-provided text prompts
(e.g., "lymphocyte", "tumor cell", "stromal cell") via cosine similarity.
Cells are assigned to the phenotype with the highest similarity score.

Approach inspired by LazySlide (MIT License).
Zheng, Y. et al. Nature Methods (2026). https://doi.org/10.1038/s41592-026-03044-7

Inputs (injected by Appose 0.10.0 -- accessed as variables, NOT task.inputs):
  tile_images: NDArray (N_tiles x H x W x C, uint8) -- batch of tile images
  phenotype_prompts: list[str] -- text prompts for each phenotype
  phenotype_names: list[str] -- display names for each phenotype
  batch_size: int -- inference batch size (default 32)

Optional inputs:
  min_similarity: float -- minimum cosine similarity for assignment (default 0.1)
  assignment_mode: str -- "argmax" (hard) or "scores" (soft) (default "argmax")

Outputs (via task.outputs):
  phenotype_labels: NDArray (N_tiles,) int32 -- assigned phenotype index (-1 = Unknown)
  similarity_scores: NDArray (N_tiles x N_phenotypes, float32) -- all similarity scores
  n_phenotypes: int -- number of phenotypes (including Unknown)
  phenotype_counts: str (JSON) -- counts per phenotype
"""
import sys
import json
import logging

logger = logging.getLogger("qpcat.zeroshot")

import numpy as np
import torch
from appose import NDArray as PyNDArray
from model_utils import detect_device

# 1. Parse inputs
tiles = tile_images.ndarray().copy()  # (N, H, W, C) uint8
n_tiles = tiles.shape[0]
prompts = list(phenotype_prompts)
names = list(phenotype_names)
n_phenotypes = len(prompts)

logger.info("Zero-shot phenotyping: %d tiles, %d phenotypes", n_tiles, n_phenotypes)
for i, (name, prompt) in enumerate(zip(names, prompts)):
    logger.info("  Phenotype %d: '%s' (prompt: '%s')", i, name, prompt)

try:
    bs = batch_size
except NameError:
    bs = 32

try:
    min_sim = min_similarity
except NameError:
    min_sim = 0.1

try:
    mode = assignment_mode
except NameError:
    mode = "argmax"

# 2. Load BiomedCLIP model (MIT License, Microsoft)
task.update("Loading BiomedCLIP model...", current=0, maximum=4)

try:
    import open_clip

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
except ImportError:
    logger.error("open_clip_torch not installed. "
                 "Add 'open_clip_torch' to pixi.toml dependencies.")
    raise

model.eval()

device = detect_device()
model = model.to(device)

# 3. Encode text prompts
task.update("Encoding text prompts...", current=1, maximum=4)

text_tokens = tokenizer(prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

logger.info("Text features encoded: %d prompts x %d dim",
            text_features.shape[0], text_features.shape[1])

# 4. Encode tile images in batches and compute similarity
task.update("Computing image-text similarities...", current=2, maximum=4)

from PIL import Image

all_similarities = []
n_batches = (n_tiles + bs - 1) // bs

for batch_idx in range(n_batches):
    start = batch_idx * bs
    end = min(start + bs, n_tiles)
    batch_tiles = tiles[start:end]

    # Preprocess tiles for BiomedCLIP
    batch_tensors = []
    for i in range(len(batch_tiles)):
        img = Image.fromarray(batch_tiles[i])
        tensor = preprocess_val(img)
        batch_tensors.append(tensor)

    batch_tensor = torch.stack(batch_tensors).to(device)

    with torch.no_grad():
        image_features = model.encode_image(batch_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Cosine similarity: (batch, n_phenotypes)
    similarities = (image_features @ text_features.T).cpu().float().numpy()
    all_similarities.append(similarities)

    if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
        logger.info("Similarity computation: batch %d/%d", batch_idx + 1, n_batches)

sim_matrix = np.concatenate(all_similarities, axis=0)  # (N_tiles, N_phenotypes)
logger.info("Similarity matrix: %s, range [%.3f, %.3f]",
            sim_matrix.shape, sim_matrix.min(), sim_matrix.max())

# 5. Assign phenotypes
task.update("Assigning phenotypes...", current=3, maximum=4)

if mode == "argmax":
    # Hard assignment: pick highest similarity, threshold by min_similarity
    best_indices = sim_matrix.argmax(axis=1)
    best_scores = sim_matrix[np.arange(n_tiles), best_indices]

    # Assign -1 (Unknown) where best score is below threshold
    labels = best_indices.astype(np.int32)
    labels[best_scores < min_sim] = -1
else:
    # Soft mode: still assign labels but also return full scores
    labels = sim_matrix.argmax(axis=1).astype(np.int32)

# Count per phenotype
counts = {}
for i, name in enumerate(names):
    counts[name] = int(np.sum(labels == i))
counts["Unknown"] = int(np.sum(labels == -1))

logger.info("Phenotype assignment complete:")
for name, count in counts.items():
    logger.info("  %s: %d (%.1f%%)", name, count, 100.0 * count / n_tiles)

# 6. Package outputs
labels_nd = PyNDArray(dtype="int32", shape=[n_tiles])
np.copyto(labels_nd.ndarray(), labels)
task.outputs["phenotype_labels"] = labels_nd

sim_nd = PyNDArray(dtype="float32", shape=list(sim_matrix.shape))
np.copyto(sim_nd.ndarray(), sim_matrix.astype(np.float32))
task.outputs["similarity_scores"] = sim_nd

task.outputs["n_phenotypes"] = n_phenotypes
task.outputs["phenotype_counts"] = json.dumps(counts)
task.outputs["phenotype_names_out"] = json.dumps(names)

logger.info("Zero-shot phenotyping results packaged")
