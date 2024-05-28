from transformers import (
    LayoutLMv3FeatureExtractor,
    LayoutLMv3TokenizerFast,
    LayoutLMv3Processor,
    LayoutLMv3ForSequenceClassification,
)
from PIL import Image, ImageDraw, ImageFont
import easyocr
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


image_folder = Path("/home/pierre/Documents/Projects/GiantsShoulder/silver_images")
image_path = image_folder / "silver2017-0.png"

reader = easyocr.Reader(["en"], gpu=True)
ocr_result = reader.readtext(str(image_path))

font_path = Path(cv2.__path__[0]) / "qt/fonts/DejaVuSansCondensed.ttf"
print("font exists:", font_path.exists())
font = ImageFont.truetype(str(font_path), size=12)


def create_bounding_box(bbox_data):
    xs = []
    ys = []
    for x, y in bbox_data:
        xs.append(x)
        ys.append(y)

    left = int(min(xs))
    top = int(min(ys))
    right = int(max(xs))
    bottom = int(max(ys))

    return [left, top, right, bottom]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 28))

left_image = Image.open(image_path).convert("RGB")
right_image = Image.new("RGB", left_image.size, (255, 255, 255))

left_draw = ImageDraw.Draw(left_image)
right_draw = ImageDraw.Draw(right_image)

for i, (bbox, word, confidence) in enumerate(ocr_result):
    box = create_bounding_box(bbox)

    left_draw.rectangle(box, outline="blue", width=2)
    left, top, right, bottom = box

    left_draw.text((right + 5, top), text=str(i + 1), fill="red", font=font)
    right_draw.text((left, top), text=word, fill="black", font=font)

ax1.imshow(left_image)
ax2.imshow(right_image)
ax1.axis("off")
ax2.axis("off")
plt.show()


from transformers import AutoModel

"/home/pierre/Data/models/layoutlmv3-base-finetuned-publaynet"
