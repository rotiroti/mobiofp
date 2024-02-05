import argparse
from pathlib import Path

import cv2
import fingerprint_enhancer
import imutils
import numpy as np
from rembg import remove
from ultralytics import YOLO

from mobiofp.api import crop_image, extract_roi, to_fingerprint


def detect(src: str, model_checkpoint: str, rotate: bool):
    image = cv2.imread(str(src))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if rotate:
        image = imutils.rotate_bound(image, 90)

    model = YOLO(
        model_checkpoint,
    )
    results = model.predict(image, conf=0.85)
    assert len(results) > 0, "No objects detected in the image."
    result = results[0]

    # Extract bounding boxes
    boxes = results[0].boxes.xyxy.tolist()
    x1, y1, x2, y2 = boxes[0]
    fingertip = image[int(y1) : int(y2), int(x1) : int(x2)]

    # Fingertip Enhancement (Grayscale conversion, Bilateral Filter, CLAHE)
    fingertip = cv2.cvtColor(fingertip, cv2.COLOR_RGB2GRAY)
    fingertip = cv2.normalize(fingertip, None, 0, 255, cv2.NORM_MINMAX)
    fingertip = cv2.bilateralFilter(fingertip, 7, 50, 50)
    fingertip = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(fingertip)

    # Use Rembg to remove background
    fingertip_mask = remove(fingertip, only_mask=True)

    # Post-process fingertip mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fingertip_mask = cv2.morphologyEx(
        fingertip_mask, cv2.MORPH_OPEN, kernel, iterations=2
    )
    fingertip_mask = cv2.GaussianBlur(
        fingertip_mask, (5, 5), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT
    )
    fingertip_mask = np.where(fingertip_mask < 127, 0, 255).astype(np.uint8)

    # Fingertip Binarization
    fingertip = cv2.bitwise_and(fingertip, fingertip, mask=fingertip_mask)
    binary = cv2.adaptiveThreshold(
        fingertip, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2
    )

    # Convert fingerphoto (fingertip) to fingerprint
    fingerprint = to_fingerprint(binary)

    # Fingerprint Enhancement
    fingerprint = fingerprint_enhancer.enhance_Fingerprint(fingerprint)

    # Save fingerprint
    PROCESSED_DIR = "./data/processed/yolo"
    fingerprint_filename = Path(str(src)).stem + ".png"
    fingerprint_filepath = PROCESSED_DIR + "/" + fingerprint_filename

    print(f"Saving fingerprint to {fingerprint_filepath}")
    _ = cv2.imwrite(fingerprint_filepath, fingerprint)


def run():
    parser = argparse.ArgumentParser(description="Segment a fingerprint from an image.")
    parser.add_argument("src", help="Path to the source image.")
    parser.add_argument("model_checkpoint", help="Path to the model checkpoint.")
    parser.add_argument(
        "--rotate", action="store_true", help="Rotate the image 90 degrees."
    )

    args = parser.parse_args()
    detect(args.src, args.model_checkpoint, args.rotate)


if __name__ == "__main__":
    run()
