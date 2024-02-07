#!/usr/bin/env python

import cv2
import argparse
import pickle

from pathlib import Path
from skimage.morphology import skeletonize

def run():
    parser = argparse.ArgumentParser(description="Extract features.")
    parser.add_argument("src", help="Path to the enhanced fingerprint image.")
    parser.add_argument("dest", help="Path to the output features file.")
    args = parser.parse_args()

    # Read image
    image = cv2.imread(args.src, cv2.IMREAD_GRAYSCALE)

    # Skeletonize the image
    image = skeletonize(image) * 255

    # Instantiate ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    keypoints, descriptors = orb.detectAndCompute(image, None)

    # Save enhanced fingerprint
    images_dir = Path(args.dest) / "features"
    images_dir.mkdir(parents=True, exist_ok=True)
    features_path = images_dir / Path(args.src).with_suffix(".pkl").name

    with open(features_path, "wb") as f:
        pickle.dump((keypoints, descriptors), f)

    print(f"Keypoints and descriptors saved to {features_path}")

if __name__ == "__main__":
    run()
