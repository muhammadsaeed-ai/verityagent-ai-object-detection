from PIL import Image
from visualization import annotate_image,save_image
def test_annotation_preserves_size_and_input():
    image=Image.new("RGB",(100,80),"white")
    result=annotate_image(image,[{"box":[10,10,60,50],"label":"person","score":.93}])
    assert result.size==image.size and image.getpixel((10,10))==(255,255,255)
def test_save_creates_directory(tmp_path):
    target=tmp_path/"nested"/"result.jpg"
    assert save_image(Image.new("RGB",(10,10)),target).is_file()
