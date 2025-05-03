# centroid_manager.py
import numpy as np
import os
import time
from config import cfg

class CentroidManager:
    def __init__(self):
        self.centroid_path = cfg.paths.DOMAIN_CENTROID_PATH
        self.metadata_path = self.centroid_path.replace('.npy', '_meta.json')
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

    def drift_since(self, old_centroid):
        centroid = self.get_centroid()
        if centroid is None or old_centroid is None:
            return None
        return float(np.linalg.norm(centroid - old_centroid))
