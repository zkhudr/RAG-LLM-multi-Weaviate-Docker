import os
import time
import json
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
from config import cfg

# Module‐level logger
_LOG = logging.getLogger(__name__)


def centroid_exists(centroid_path: str) -> bool:
    """Return True if the given centroid .npy file exists."""
    return Path(centroid_path).exists()


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 − cosine similarity between two vectors."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def should_recalculate_centroid(
    new_vectors: List[np.ndarray],
    all_vectors: List[np.ndarray],
    old_centroid: Optional[np.ndarray],
    threshold: float = 0.05,
    diversity_threshold: float = 0.01
) -> bool:
    """
    Decide if centroid should be recalculated:
    - No prior centroid or no vectors → recalc.
    - Fraction of new vectors ≥ threshold → recalc.
    - Centroid drift ≥ diversity_threshold → recalc.
    """
    if not all_vectors:
        return True

    percent_new = len(new_vectors) / len(all_vectors)
    if percent_new >= threshold:
        return True

    new_centroid = np.mean(np.vstack(all_vectors), axis=0)
    if old_centroid is not None:
        dist = cosine_distance(old_centroid, new_centroid)
        if dist >= diversity_threshold:
            return True

    return False


def save_metadata(self, stats: dict) -> None:
        """Save descriptive stats (not just shape info) to metadata file."""
        try:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            stats["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            self.logger.info(f"Saved descriptive centroid metadata to {self.metadata_path}")
        except Exception as e:
            self.logger.error(f"Failed saving descriptive metadata: {e}")


def get_centroid_stats(centroid: np.ndarray) -> dict:
    """
    Returns descriptive statistics for a centroid vector.
    """
    return {
        "Dimension Count":     centroid.size,
        "Mean":                float(np.mean(centroid)),
        "Std Dev":             float(np.std(centroid)),
        "Min":                 float(np.min(centroid)),
        "25th Percentile":     float(np.percentile(centroid, 25)),
        "Median":              float(np.median(centroid)),
        "75th Percentile":     float(np.percentile(centroid, 75)),
        "Max":                 float(np.max(centroid)),
        "L2 Norm (Magnitude)": float(np.linalg.norm(centroid)),
    }


class CentroidManager:
    def __init__(
        self,
        centroid_path: str    = None,
        collection_name: str  = None,
        base_path: str        = None
    ):
        self.logger = _LOG

        # Determine file path for .npy
        if centroid_path and not (collection_name or base_path):
            p = Path(centroid_path)
        else:
            # Base directory for all centroids
            base_dir = (
                base_path
                or getattr(cfg.paths, 'CENTROID_DIR', None)
                or os.path.dirname(getattr(cfg.paths, 'DOMAIN_CENTROID_PATH', ''))
                or '.'
            )
            Path(base_dir).mkdir(parents=True, exist_ok=True)

            if collection_name:
                fname = f"{collection_name}_centroid.npy"
                p = Path(base_dir) / fname
            elif centroid_path:
                p = Path(centroid_path)
            else:
                p = Path(getattr(cfg.paths, 'DOMAIN_CENTROID_PATH', './domain_centroid.npy'))

        self.centroid_path = p
        self.metadata_path = self.centroid_path.with_name(self.centroid_path.stem + "_meta.json")

        self.centroid: Optional[np.ndarray] = None
        self._warned_missing = False
        self._load_centroid()

    def _load_centroid(self) -> None:
        """Load centroid from disk if available; else leave as None."""
        if self.centroid_path.exists():
            try:
                self.centroid = np.load(self.centroid_path)
            except Exception as e:
                self.logger.error(f"⚠️ Failed to load centroid: {e}")
                self.centroid = None
        else:
            self.centroid = None

    def save_centroid(self, vector: np.ndarray) -> None:
        """Save centroid array and record metadata (.npy + _meta.json)."""
        try:
            self.centroid_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.centroid_path, vector)
            self.centroid = vector

            meta = {
                "shape": vector.shape,
                "dtype": str(vector.dtype),
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            self.logger.info(
                f"Saved centroid to {self.centroid_path} and metadata to {self.metadata_path}"
            )
        except Exception as e:
            self.logger.error(f"⚠️ Failed to save centroid: {e}")

    def save_metadata(self, stats: dict) -> None:
            """Save descriptive stats (not just shape info) to metadata file."""
            try:
                self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
                stats["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                with open(self.metadata_path, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
                self.logger.info(f"Saved descriptive centroid metadata to {self.metadata_path}")
            except Exception as e:
                self.logger.error(f"Failed saving descriptive metadata: {e}")


    def get_centroid(self) -> Optional[np.ndarray]:
        """
        Return the centroid array if it exists, else None.
        Logs a warning once if missing.
        """
        if self.centroid is not None:
            return self.centroid

        if self.centroid_path.exists():
            try:
                self.centroid = np.load(self.centroid_path)
                return self.centroid
            except Exception as e:
                self.logger.error(f"Failed loading centroid: {e}")
                return None

        if not self._warned_missing:
            self.logger.warning(f"Centroid file not found at {self.centroid_path}")
            self._warned_missing = True
        return None

    def get_metadata(self) -> dict:
        """Return stored metadata or empty dict."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed loading metadata: {e}")
        return {}





    def query_insight(self, query_vector: np.ndarray) -> dict:
        """
        Compare a query vector to the centroid:
        Returns distance & cosine similarity, or an error if no centroid.
        """
        centroid = self.get_centroid()
        if centroid is None:
            return {"error": "Centroid not calculated yet."}
        dist = float(np.linalg.norm(query_vector - centroid))
        sim = float(np.dot(query_vector, centroid) /
                    (np.linalg.norm(query_vector) * np.linalg.norm(centroid)))
        return {"distance": dist, "similarity": sim}

    def get_all_vectors(self, client, collection_name: str) -> List[np.ndarray]:
        """Fetch all vectors from a Weaviate collection as numpy arrays."""
        vectors: List[np.ndarray] = []
        try:
            collection = client.collections.get(collection_name)
        except Exception as e:
            self.logger.error(f"Collection '{collection_name}' fetch failed: {e}")
            return vectors

        for obj in collection.iterator(include_vector=True, return_properties=[]):
            vec = obj.vector.get('default')
            if vec:
                vectors.append(np.array(vec))
        return vectors

    def drift_since(self, old_centroid: Optional[np.ndarray]) -> Optional[float]:
        """Compute L2 drift since a previous centroid; or None if unavailable."""
        centroid = self.get_centroid()
        if centroid is None or old_centroid is None:
            return None
        return float(np.linalg.norm(centroid - old_centroid))



