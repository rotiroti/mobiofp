import argparse
from pathlib import Path

import cv2
import fingerprint_enhancer
import imutils

from mobiofp.api import crop_image, extract_roi, to_fingerprint
from mobiofp.unet import Segment


def segment(src: str, model_checkpoint: str, rotate: bool):
    # Read RGB sample image
    image = cv2.imread(str(src))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if rotate:
        image = imutils.rotate_bound(image, 90)

    # Load a U-Net pre-trained model
    model = Segment()
    model.load(model_checkpoint)
    mask = model.predict(image)

    # Fingertip ROI extraction
    bbox = extract_roi(mask)
    fingertip = crop_image(image, bbox)
    fingertip_mask = crop_image(mask, bbox)

    # Fingertip Enhancement (Grayscale conversion, Bilateral Filter, CLAHE)
    fingertip = cv2.cvtColor(fingertip, cv2.COLOR_RGB2GRAY)
    fingertip = cv2.normalize(fingertip, None, 0, 255, cv2.NORM_MINMAX)
    fingertip = cv2.bilateralFilter(fingertip, 7, 50, 50)
    fingertip = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(fingertip)

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
    PROCESSED_DIR = "./data/processed/unet"
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
    segment(args.src, args.model_checkpoint, args.rotate)


if __name__ == "__main__":
    segment("data/raw/unet/1.png", "model/unet.h5", False)
