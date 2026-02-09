import numpy as np
from typing import List, Tuple
from insightface.app import FaceAnalysis

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

class FaceEngine:
    def __init__(self, model_name: str = "buffalo_l"):
        self.app = FaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def detect_faces(self, frame_bgr: np.ndarray):
        """
        Returns list of insightface Face objects (each has bbox, det_score, embedding, kps, etc.)
        """
        return self.app.get(frame_bgr)

    def recognize_one(
        self,
        embedding: np.ndarray,
        known: List[Tuple[str, np.ndarray]],
        threshold: float
    ) -> Tuple[str, float]:
        """
        Match against all known embeddings. Returns best (name, score).
        name='UNKNOWN' if best_score < threshold.
        """
        if not known:
            return ("UNKNOWN", 0.0)

        best_name = "UNKNOWN"
        best_score = -1.0

        for name, emb in known:
            score = cosine_similarity(embedding, emb)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= threshold:
            return (best_name, best_score)
        return ("UNKNOWN", best_score)
