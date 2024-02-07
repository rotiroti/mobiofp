#!/usr/bin/env python

import cv2
import argparse

from pathlib import Path
from mobiofp.utils import to_fingerprint, enhance_fingerprint

def run():
    parser = argparse.ArgumentParser(description="Fingerphoto enhancement.")
    parser.add_argument("src", help="Path to the input fingertip image.")
    parser.add_argument("mask", help="Path to the input fingertip mask.")
    parser.add_argument("dest", help="Path to the output image.")
    args = parser.parse_args()

    # Read fingertip and mask images (Grayscale)
    image = cv2.imread(args.src, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

    fingertip = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    fingertip = cv2.bilateralFilter(image, 7, 50, 50)
    fingertip = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)

    # Fingertip Binarization
    fingertip = cv2.bitwise_and(fingertip, fingertip, mask=mask)
    binary = cv2.adaptiveThreshold(fingertip, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2)

    # Convert fingerphoto (fingertip) to fingerprint
    fingerprint = to_fingerprint(binary)

    # Fingerprint Enhancement
    fingerprint = enhance_fingerprint(fingerprint)
    fingerprint = fingerprint.astype("uint8")

    # Save enhanced fingerprint
    images_dir = Path(args.dest) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    fingerprint_path = images_dir / Path(args.src).with_suffix(".png").name
    cv2.imwrite(str(fingerprint_path), fingerprint)

    print(f"Enhanced fingerprint saved to {fingerprint_path}")

if __name__ == "__main__":
    run()
