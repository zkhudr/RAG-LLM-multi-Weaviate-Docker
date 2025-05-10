# centroid_manager.py
import numpy as np
import os
import time
from config import cfg


def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def should_recalculate_centroid(new_vectors, all_vectors, old_centroid, threshold=0.05, diversity_threshold=0.01):
    """
    Decide if centroid should be recalculated.
    - new_vectors: list of np arrays (new/changed)
    - all_vectors: list of np arrays (all)
    - old_centroid: np array (previous centroid)
    - threshold: fraction (e.g., 0.05 for 5%)
    - diversity_threshold: min cosine distance for centroid drift
    """
    if not all_vectors or len(all_vectors) == 0:
        return True  # Always recalculate if nothing existed before

    percent_new = len(new_vectors) / len(all_vectors)
    if percent_new >= threshold:
        return True

    new_centroid = np.mean(np.vstack(all_vectors), axis=0)
    if old_centroid is not None:
        dist = cosine_distance(old_centroid, new_centroid)
        if dist >= diversity_threshold:
            return True

    return False


class CentroidManager:
    def __init__(self):
    # Handle case where cfg might be None
        from config import cfg
        if cfg and hasattr(cfg, 'paths'):
            self.centroid_path = cfg.paths.DOMAIN_CENTROID_PATH
            self.metadata_path = self.centroid_path.replace('.npy', '_meta.json')
        else:
            # Default fallback paths
            self.centroid_path = "./domain_centroid.npy"
            self.metadata_path = "./domain_centroid_meta.json"
        self._load_centroid()

    def _load_centroid(self):
        if os.path.exists(self.centroid_path):
            self.centroid = np.load(self.centroid_path)
        else:
            self.centroid = None

    def save_centroid(self, vector):
        np.save(self.centroid_path, vector)
        self.centroid = vector
        # Save metadata
        meta = {"updated_at": time.time(), "shape": vector.shape}
        with open(self.metadata_path, "w") as f:
            import json; json.dump(meta, f)

    def get_centroid(self):
        if self.centroid is None and os.path.exists(self.centroid_path):
            self.centroid = np.load(self.centroid_path)
        return self.centroid

    def get_metadata(self):
        if os.path.exists(self.metadata_path):
            import json
            with open(self.metadata_path) as f:
                return json.load(f)
        return {}

    def query_insight(self, query_vector):
        centroid = self.get_centroid()
        if centroid is None:
            return {"error": "Centroid not calculated yet."}
        dist = float(np.linalg.norm(query_vector - centroid))
        sim = float(np.dot(query_vector, centroid) / (np.linalg.norm(query_vector) * np.linalg.norm(centroid)))
        return {"distance": dist, "similarity": sim}

    def get_all_vectors(self, client, collection_name):
        """Fetch all vectors from Weaviate collection as a list of np arrays."""
        import numpy as np
        vectors = []
        collection = client.collections.get(collection_name)
        for obj in collection.iterator(include_vector=True, return_properties=[]):
            if obj.vector and 'default' in obj.vector:
                vectors.append(np.array(obj.vector['default']))
        return vectors
    
    def drift_since(self, old_centroid):
        centroid = self.get_centroid()
        if centroid is None or old_centroid is None:
            return None
        return float(np.linalg.norm(centroid - old_centroid))
    
   
        return False
