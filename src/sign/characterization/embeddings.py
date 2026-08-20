"""Sentence embeddings + dimensionality reduction for Phase 2's
"do the two datasets occupy different regions of representation space"
question. CPU-only (no VM needed) -- `all-MiniLM-L6-v2` is small enough
to embed a few thousand short sentences in minutes on a laptop CPU.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def embed_texts(texts: list[str], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    return model.encode(texts, show_progress_bar=False, batch_size=64)


def reduce_dimensions(embeddings: np.ndarray, seed: int = 42) -> dict[str, np.ndarray]:
    """2D PCA and 2D UMAP projections of the same embedding matrix."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=seed)
    pca_coords = pca.fit_transform(embeddings)

    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=seed)
        umap_coords = reducer.fit_transform(embeddings)
    except Exception:
        umap_coords = None

    return {"pca": pca_coords, "umap": umap_coords, "pca_explained_variance_ratio": pca.explained_variance_ratio_}


def sample_for_embedding(df: pd.DataFrame, text_col: str, group_col: str, n_per_group: int, seed: int) -> pd.DataFrame:
    """Deterministic, equal-sized-per-group sample, so the PCA/UMAP plot
    isn't visually dominated by whichever group happens to be largest
    (SIGN interpretations has ~15,000 rows vs. Dataset A's 9,386 vs. SIGN
    originals' ~2,827)."""
    parts = []
    for group, sub in df.groupby(group_col):
        k = min(n_per_group, len(sub))
        parts.append(sub.sample(k, random_state=seed))
    return pd.concat(parts, ignore_index=True)
