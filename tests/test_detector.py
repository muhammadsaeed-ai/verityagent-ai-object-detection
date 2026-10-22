import pytest
import torch
from detector import filter_detections, get_label_map

def pred(boxes, labels, scores):
    return {"boxes":torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),"labels":torch.tensor(labels,dtype=torch.int64),"scores":torch.tensor(scores,dtype=torch.float32)}
def test_confidence_filter_keeps_threshold_and_higher():
    got=filter_detections(pred([[1,2,10,20],[2,3,8,9],[0,0,1,1]],[1,2,3],[.49,.50,.91]),.50)
    assert [d["label_id"] for d in got]==[2,3] and got[0]["box"]==[2,3,8,9]
def test_empty_after_filter(): assert filter_detections(pred([[0,0,1,1]],[1],[.1]),.5)==[]
def test_invalid_threshold():
    with pytest.raises(ValueError,match="between 0.0 and 1.0"): filter_detections(pred([],[],[]),1.1)
def test_label_map_uses_all_categories():
    names=["__background__","person","bicycle","car"]
    assert get_label_map(names)==dict(enumerate(names))
def test_mismatched_lengths():
    with pytest.raises(ValueError,match="equal lengths"): filter_detections(pred([[0,0,1,1]],[1,2],[.9]),.5)
def test_missing_keys():
    with pytest.raises(ValueError,match="must contain"): filter_detections({"scores":torch.tensor([.9])},.5)

from detector import load_image

def test_load_image_missing_path_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_image(tmp_path / "missing.jpg")

def test_load_image_rejects_invalid_file(tmp_path):
    invalid = tmp_path / "broken.jpg"
    invalid.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not read a valid image"):
        load_image(invalid)
