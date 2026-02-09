from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    # Where embeddings DB is stored
    db_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faces.sqlite")

    # Recognition threshold (cosine similarity). Higher = stricter.
    # Typical good range: 0.35 - 0.55 depending on lighting/camera.
    match_threshold: float = 0.45

    # Camera index (0 is default webcam)
    camera_index: int = 0

    # InsightFace model name
    model_name: str = "buffalo_l"

settings = Settings()
