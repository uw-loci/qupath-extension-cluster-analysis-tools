"""
Shared utilities for QP-CAT foundation model scripts.

Provides device detection and the foundation model registry.
Used by extract_features.py and zero_shot_phenotyping.py.

Foundation model integration inspired by LazySlide (MIT License).
Zheng, Y. et al. Nature Methods (2026). https://doi.org/10.1038/s41592-026-03044-7
"""
import logging
import torch

logger = logging.getLogger("qpcat.model_utils")

# Foundation model registry: name -> (timm_id, embed_dim, license)
# Only models with commercially-permissive licenses (Apache 2.0, MIT).
# Models are downloaded on-demand from HuggingFace, not bundled.
FOUNDATION_MODELS = {
    "h-optimus-0": ("hf_hub:bioptimus/H-optimus-0", 1536, "Apache 2.0"),
    "virchow": ("hf_hub:paige-ai/Virchow", 2560, "Apache 2.0"),
    "hibou-l": ("hf_hub:histai/hibou-L", 1024, "Apache 2.0"),
    "hibou-b": ("hf_hub:histai/hibou-b", 768, "Apache 2.0"),
    "midnight": ("hf_hub:kaiko-ai/midnight", 1536, "MIT"),
    "dinov2-large": ("hf_hub:facebook/dinov2-large", 1024, "Apache 2.0"),
}


def detect_device():
    """Detect the best available compute device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        logger.info("Using CUDA GPU")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using Apple MPS")
        return "mps"
    else:
        logger.info("Using CPU")
        return "cpu"
