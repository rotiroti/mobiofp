#!/usr/bin/env python

import cv2
import argparse
import fingerprint_feature_extractor as ffe
import numpy as np

from pathlib import Path

def run():
    parser = argparse.ArgumentParser(description="Extract minutiae.")
    parser.add_argument("source_file", help="Path to the enhanced fingerprint image.")
    parser.add_argument("target_file", help="Path to the output minutiae file.")
    parser.add_argument("--threshold", help="Threshold for false minutiae", type=int, default=10)
    args = parser.parse_args()

    # Read image
    image = cv2.imread(args.source_file, cv2.IMREAD_GRAYSCALE)

    # Extract minutiae features
    terminations, bifurcations = ffe.extract_minutiae_features(image, args.threshold)
    minutiae = np.array([[int(minutia.locX), int(minutia.locY), float(minutia.Orientation[0]), 1 if minutia.Type == 'Termination' else 3] for minutia in terminations + bifurcations])

    # Save enhanced fingerprint
    images_dir = Path(args.target_file) / "minutiae"
    images_dir.mkdir(parents=True, exist_ok=True)
    minutiae_path = images_dir / Path(args.source_file).with_suffix(".txt").name

    np.savetxt(str(minutiae_path), minutiae, fmt=['%d', '%d', '%.2f', '%d'])

    print(f"Minutiae saved to {minutiae_path}")

if __name__ == "__main__":
    run()
