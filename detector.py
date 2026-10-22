"""Faster R-CNN inference and prediction post-processing."""
from pathlib import Path
from typing import Any, Sequence
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

def load_image(image_path: str | Path) -> Image.Image:
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Input image does not exist or is not a file: {path}")
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not read a valid image from '{path}': {exc}") from exc

def load_model(device: torch.device | None = None) -> tuple[torch.nn.Module, Any, torch.device]:
    selected = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    try:
        model = fasterrcnn_resnet50_fpn(weights=weights)
        model.to(selected)
        model.eval()
    except Exception as exc:
        raise RuntimeError("Failed to load Faster R-CNN. Check compatible PyTorch/Torchvision versions and internet access for the first-time weights download.") from exc
    return model, weights, selected

def get_label_map(categories: Sequence[str]) -> dict[int, str]:
    """Map every model class index using official weights metadata."""
    return dict(enumerate(categories))

def filter_detections(prediction: dict[str, torch.Tensor], confidence_threshold: float) -> list[dict[str, Any]]:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0.")
    boxes, labels, scores = (prediction.get(k) for k in ("boxes", "labels", "scores"))
    if boxes is None or labels is None or scores is None:
        raise ValueError("Prediction must contain 'boxes', 'labels', and 'scores'.")
    if not (len(boxes) == len(labels) == len(scores)):
        raise ValueError("Prediction boxes, labels, and scores must have equal lengths.")
    result = []
    for box, label, score in zip(boxes, labels, scores):
        confidence = float(score.item())
        if confidence >= confidence_threshold:
            result.append({"box": [int(round(float(v))) for v in box.tolist()], "label_id": int(label.item()), "score": confidence})
    return result

def detect_objects(image: Image.Image, model: torch.nn.Module, weights: Any, device: torch.device, confidence_threshold: float = .50) -> tuple[list[dict[str, Any]], dict[int, str]]:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0.")
    tensor = weights.transforms()(image.convert("RGB")).to(device)
    with torch.inference_mode():
        predictions = model([tensor])
    if not predictions:
        raise RuntimeError("The detection model returned no prediction.")
    detections = filter_detections(predictions[0], confidence_threshold)
    categories = weights.meta.get("categories")
    if not categories:
        raise RuntimeError("The selected model weights do not expose category metadata.")
    labels = get_label_map(categories)
    for item in detections:
        item["label"] = labels.get(item["label_id"], f"class_{item['label_id']}")
    return detections, labels
