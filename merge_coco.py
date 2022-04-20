import json

import numpy as np
from PIL import ImageDraw, ImageFont
from PIL import Image

import service
import argparse
import os
import cv2

from core.element.TextImg import FreeTypeFontOffset
from service import SmoothAreaProvider
from string import ascii_lowercase, ascii_uppercase, digits


def merge(dir_src, json_path):
    print(dir_src)
    filenames = os.listdir(dir_src)
    filenames.sort()

    with open(json_path, "w") as f:
        load_dict = {"categories": list(), "images": list(), "annotations": list()}
        load_dict["categories"].append({"id": 1, "name": "text", "supercategory": "text"})

        # Character level segmentation
        base_offset = 2
        kv_digits = {digits[idx]: idx + base_offset for idx in range(len(digits))}
        kv_upper = {ascii_uppercase[idx]: idx + base_offset + len(kv_digits) for idx in range(len(ascii_uppercase))}
        kv_lower = {
            ascii_lowercase[idx]: idx + base_offset + len(kv_digits) + len(kv_upper)
            for idx in range(len(ascii_lowercase))
        }
        category_map = {**kv_digits, **kv_upper, **kv_lower}

        for k in category_map:
            v = category_map[k]
            load_dict["categories"].append({"id": v, "name": k, "supercategory": "text"})

        for filename in filenames:
            try:
                single_json_path = os.path.join(dir_src, filename)
                print(single_json_path)

                with open(single_json_path, "r") as sf:
                    text = sf.read()
                    single = json.loads(text)
                    s_images = single["images"]
                    s_annotations = single["annotations"]

                    for idx in range(len(s_images)):
                        load_dict["images"].append(s_images[idx])

                    for idx in range(len(s_annotations)):
                        load_dict["annotations"].append(s_annotations[idx])

            except Exception as e:
                print(e)

        json.dump(load_dict, f, indent=2)
        # json.dump(load_dict, f)


if __name__ == "__main__":
    merge("./output/coco_data/json", "./output/coco_data/train.json")
