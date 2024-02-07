#!/usr/bin/env python

import cv2
import imutils
import argparse

from pathlib import Path
from mobiofp.segmentation import Segment
from mobiofp.utils import extract_roi, crop_image

def run():
    parser = argparse.ArgumentParser(description="Fingertip segmentation.")
    parser.add_argument("src", help="Path to the input image.")
    parser.add_argument("model", help="Path to the model checkpoint.")
    parser.add_argument("--dest", help="Path to the output image.", type=str, default="./data/processed/segmentation/fingertip")
    args = parser.parse_args()

    # Read RGB sample image
    image = cv2.imread(args.src)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = imutils.rotate_bound(image, 90)

    # Load segmentation model
    segmenter = Segment()
    segmenter.load(args.model)
    mask = segmenter.predict(image)

    # Fingertip ROI extraction
    bbox = extract_roi(mask)
    fingertip = crop_image(image, bbox)
    fingertip_mask = crop_image(mask, bbox)

    # Normalize final image and mask
    fingertip = cv2.normalize(fingertip, None, 0, 255, cv2.NORM_MINMAX)
    fingertip = cv2.cvtColor(fingertip, cv2.COLOR_RGB2BGR)
    fingertip_mask = fingertip_mask * 255

    # Save fingertip and mask
    images_dir = Path(args.dest) / "images"
    labels_dir = Path(args.dest) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    fingertip_path = images_dir / Path(args.src).name
    cv2.imwrite(str(fingertip_path), fingertip)
    print(f"Fingertip image saved to {fingertip_path}")

    fingertip_mask_path = labels_dir / Path(args.src).with_suffix(".png").name
    cv2.imwrite(str(fingertip_mask_path), fingertip_mask)
    print(f"Fingertip mask saved to {fingertip_mask_path}")

if __name__ == "__main__":
    run()
