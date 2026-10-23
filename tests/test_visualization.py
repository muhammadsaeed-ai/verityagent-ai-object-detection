import cv2
import pytest
from PIL import Image
from visualization import annotate_image, display_image, save_image
def test_annotation_preserves_size_and_input():
    image=Image.new("RGB",(100,80),"white")
    result=annotate_image(image,[{"box":[10,10,60,50],"label":"person","score":.93}])
    assert result.size==image.size and image.getpixel((10,10))==(255,255,255)
def test_save_creates_directory(tmp_path):
    target=tmp_path/"nested"/"result.jpg"
    assert save_image(Image.new("RGB",(10,10)),target).is_file()


def test_display_failure_is_not_masked_by_cleanup(monkeypatch):
    def fail_display(*_args, **_kwargs):
        raise cv2.error("GUI support unavailable")

    def fail_cleanup(*_args, **_kwargs):
        raise cv2.error("window cleanup unavailable")

    monkeypatch.setattr(cv2, "imshow", fail_display)
    monkeypatch.setattr(cv2, "destroyAllWindows", fail_cleanup)
    with pytest.raises(RuntimeError, match="Image display failed"):
        display_image(Image.new("RGB", (8, 8)))
