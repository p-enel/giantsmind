# importing prerequisites
import sys
import requests
import tarfile
import json
import numpy as np
from os import path
from PIL import Image
from PIL import ImageFont, ImageDraw
from glob import glob
from matplotlib import pyplot as plt


colors = {
    "title": (255, 0, 0),
    "text": (0, 255, 0),
    "figure": (0, 0, 255),
    "table": (255, 255, 0),
    "list": (0, 255, 255),
}


# Function to viz the annotation
def markup(image, annotations, samples, font):
    """Draws the segmentation, bounding box, and label of each annotation"""
    draw = ImageDraw.Draw(image, "RGBA")
    for annotation in annotations:
        # Draw segmentation
        draw.polygon(
            annotation["segmentation"][0],
            fill=colors[samples["categories"][annotation["category_id"] - 1]["name"]]
            + (64,),
        )
        # Draw bbox
        draw.rectangle(
            (
                annotation["bbox"][0],
                annotation["bbox"][1],
                annotation["bbox"][0] + annotation["bbox"][2],
                annotation["bbox"][1] + annotation["bbox"][3],
            ),
            outline=colors[samples["categories"][annotation["category_id"] - 1]["name"]]
            + (255,),
            width=2,
        )
        # Draw label
        bbox = draw.textbbox(
            (annotation["bbox"][0] + annotation["bbox"][2], annotation["bbox"][1]),
            text=samples["categories"][annotation["category_id"] - 1]["name"],
            font=font,
        )

        if annotation["bbox"][3] < bbox[3] - bbox[1]:
            draw.rectangle(bbox, fill=(64, 64, 64, 255))
            draw.text(
                (annotation["bbox"][0] + annotation["bbox"][2], annotation["bbox"][1]),
                text=samples["categories"][annotation["category_id"] - 1]["name"],
                fill=(255, 255, 255, 255),
                font=font,
            )
        else:
            draw.rectangle(bbox, fill=(64, 64, 64, 255))
            draw.text(
                (annotation["bbox"][0], annotation["bbox"][1]),
                text=samples["categories"][annotation["category_id"] - 1]["name"],
                fill=(255, 255, 255, 255),
                font=font,
            )
    return np.array(image)


fname = "examples.tar.gz"
url = "https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/" + fname
r = requests.get(url)
open(fname, "wb").write(r.content)

# Extracting the dataset
tar = tarfile.open(fname)
tar.extractall()
tar.close()

# Verifying the file was extracted properly
data_path = "examples/"
path.exists(data_path)

# Parse the JSON file and read all the images and labels
with open("examples/samples.json", "r") as fp:
    samples = json.load(fp)
# Index images
images = {}
for image in samples["images"]:
    images[image["id"]] = {
        "file_name": "examples/" + image["file_name"],
        "annotations": [],
    }
for ann in samples["annotations"]:
    images[ann["image_id"]]["annotations"].append(ann)


# Visualize annotations
font = ImageFont.truetype("examples/DejaVuSans.ttf", 15)
fig = plt.figure(figsize=(16, 100))
for i, (_, image) in enumerate(images.items()):
    with Image.open(image["file_name"]) as img:
        ax = plt.subplot(int(len(images)) // 2, 2, i + 1)
        ax.imshow(markup(img, image["annotations"], samples, font))
        ax.axis("off")
plt.subplots_adjust(hspace=0, wspace=0)
