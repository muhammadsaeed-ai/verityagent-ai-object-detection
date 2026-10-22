import pytest
from config import DetectionConfig
def test_defaults():
    c=DetectionConfig()
    assert str(c.image_path)=="assets/input.jpg" and str(c.output_path)=="output/annotated.jpg"
    assert c.confidence_threshold==.50 and c.display is True
def test_invalid_threshold():
    with pytest.raises(ValueError,match="between 0.0 and 1.0"): DetectionConfig(confidence_threshold=-.1)
