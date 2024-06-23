import os
import json
import cv2
import numpy as np
from utils import scale_polygon, upscale_image
from utils import mask2polygon, get_area, get_bbox, normalize_polygons, get_bbox_from_mask
from label_studio_converter import brush

def process_images(input_folder, output_json):
    data = {
        "info": {
            "year": "2024",
            "version": "2",
            "description": "Mentally ill",
            "contributor": "Mert",
            "url": "url",
            "date_created": "2024-06-14T00:00:00+00:00"
        },
        "licenses": [
            {
                "id": 1,
                "url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "name": "Public Domain"
            }
        ],
        "categories": [
            {
                "id": 0,
                "name": "Cars",
                "supercategory": "none"
            },
            {
                "id": 1,
                "name": "car",
                "supercategory": "cars"
            }
        ],
        "images": [],
        "annotations": []
    }

    image_id = 0
    annotation_id = 0

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            print(f"Processing {filename}")
            filepath = os.path.join(input_folder, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            height, width = image.shape[:2]

            _, binary_mask = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
            mask = binary_mask.astype(np.uint8)*255
            area_pixels = cv2.countNonZero(binary_mask)

            bbox = get_bbox_from_mask(binary_mask)

            rle = brush.mask2rle(mask)

            data["images"].append({
                "id": image_id,
                "license": 1,
                "file_name": filename.replace(".png", ".jpg"),
                "height": height,
                "width": width,
                "date_captured": "2020-07-20T19:39:26+00:00"
            })
            
            data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": rle,
                "area": area_pixels,
                "bbox": bbox,
                "iscrowd": 0
            })

            image_id += 1
            annotation_id += 1

    with open(output_json, 'w') as outfile:
        json.dump(data, outfile, indent=4)

if __name__ == "__main__":
    input_folder = "/Users/mert/Downloads/CocoDengeliSegData/val"
    output_json = "/Users/mert/Downloads/CocoDengeliSegData/ann/coco_val_ann.json"

    process_images(input_folder, output_json)
