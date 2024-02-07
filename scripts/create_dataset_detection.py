#!/usr/bin/env python

import argparse

from mobiofp.detection import UltralyticsDataset

def run():
    """
    Runs the script to create a dataset for YOLO object detection.

    This script takes in the paths to the images directory, labels directory, and the output directory for the generated dataset.
    It also accepts an optional argument --train-ratio to specify the ratio of images to be included in the training set.

    Example usage:
    python yolo_detection_dataset.py images_dir labels_dir output_dir --train-ratio 0.8
    """
    parser = argparse.ArgumentParser(description="Create dataset for YOLO object detection.")
    parser.add_argument("images_dir", type=str, help="Name of the images directory.")
    parser.add_argument("labels_dir", type=str, help="Name of the labels directory.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for the generated dataset.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of images to be included in the training set.")
    args = parser.parse_args()

    dataset = UltralyticsDataset(args.images_dir, args.labels_dir, args.output_dir)
    dataset.create(args.train_ratio)

if __name__ == "__main__":
    run()
