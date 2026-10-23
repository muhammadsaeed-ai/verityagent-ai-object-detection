"""Image annotation, saving, and optional GUI display."""
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image

def annotate_image(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    canvas = cv2.cvtColor(np.asarray(image.convert("RGB")).copy(), cv2.COLOR_RGB2BGR)
    h, w = canvas.shape[:2]
    for item in detections:
        x1,y1,x2,y2 = map(int, item["box"])
        x1,x2 = max(0,min(w-1,x1)),max(0,min(w-1,x2))
        y1,y2 = max(0,min(h-1,y1)),max(0,min(h-1,y2))
        if x2 < x1 or y2 < y1: continue
        caption = f'{item.get("label", "object")} {float(item["score"]):.2f}'
        color=(255,190,0)
        cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
        (tw,th),base=cv2.getTextSize(caption,cv2.FONT_HERSHEY_SIMPLEX,.55,2)
        ty=max(th+6,y1)
        cv2.rectangle(canvas,(x1,max(0,ty-th-6)),(min(w-1,x1+tw+8),min(h-1,ty+base)),color,cv2.FILLED)
        cv2.putText(canvas,caption,(x1+4,max(th,ty-3)),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),2,cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))

def save_image(image: Image.Image, output_path: str | Path) -> Path:
    path=Path(output_path).expanduser()
    path.parent.mkdir(parents=True,exist_ok=True)
    try: image.save(path)
    except OSError as exc: raise OSError(f"Could not save annotated image to '{path}': {exc}") from exc
    return path

def display_image(image: Image.Image, window_name: str = "VerityAgent Object Detection") -> None:
    try:
        cv2.imshow(window_name,cv2.cvtColor(np.asarray(image.convert("RGB")),cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
    except cv2.error as exc:
        raise RuntimeError("Image display failed. Use --no-display on headless systems or install OpenCV with GUI support.") from exc
    finally:
        # Cleanup must not mask the original GUI error (common on headless builds).
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
