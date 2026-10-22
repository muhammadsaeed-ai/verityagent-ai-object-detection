"""Default settings for the VerityAgent object detector."""
from dataclasses import dataclass
from pathlib import Path
@dataclass(frozen=True)
class DetectionConfig:
    image_path: Path = Path("assets/input.jpg")
    output_path: Path = Path("output/annotated.jpg")
    confidence_threshold: float = 0.50
    display: bool = True
    def __post_init__(self):
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0.")
