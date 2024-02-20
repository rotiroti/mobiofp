import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import typer
from rembg import new_session, remove
from tqdm import tqdm
from ultralytics import YOLO, settings

from mobiofp.segmentation import Segment
from mobiofp.utils import (
    crop_image,
    extract_roi,
    fingerprint_enhancement,
    fingerprint_mapping,
    fingertip_enhancement,
    fingertip_thresholding,
    post_process_mask,
    quality_scores,
)

app = typer.Typer()


@app.command(help="Run fingertip segmentation using a custom U-Net model.")
def segment(
    source_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    model_file: Path = typer.Argument(..., help="Path to the custom custom U-Net model weights."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
    factor: float = typer.Option(1.0, help="Region of interest factor."),
):
    # Load segmentation model
    segmenter = Segment()
    segmenter.load(str(model_file))

    # Create output directories
    images_dir = Path(target_directory) / "fingertips"
    masks_dir = Path(target_directory) / "masks"
    bbox_dir = Path(target_directory) / "bbox"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    bbox_dir.mkdir(parents=True, exist_ok=True)

    # Process each image in the directory
    for image_path in tqdm(list(Path(source_directory).glob("*.jpg"))):
        # Read RGB image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Predict mask
        result = segmenter.predict(image)
        bbox = extract_roi(result, factor)
        fingertip = crop_image(image, bbox)
        fingertip_mask = crop_image(result, bbox)

        # Normalize final image and mask
        fingertip = cv2.normalize(fingertip, None, 0, 255, cv2.NORM_MINMAX)
        fingertip = cv2.cvtColor(fingertip, cv2.COLOR_RGB2BGR)
        fingertip_mask = fingertip_mask * 255

        # Save fingertip, mask and bbox
        fingertip_path = images_dir / image_path.name
        cv2.imwrite(str(fingertip_path), fingertip)
        typer.echo(f"Fingertip image saved to {fingertip_path}")

        fingertip_mask_path = masks_dir / image_path.with_suffix(".png").name
        cv2.imwrite(str(fingertip_mask_path), fingertip_mask)
        typer.echo(f"Fingertip mask saved to {fingertip_mask_path}")

        bbox_array = np.array(bbox).reshape(1, 4)
        fingertip_bbox_path = bbox_dir / image_path.with_suffix(".txt").name
        np.savetxt(str(fingertip_bbox_path), bbox_array, fmt="%d")
        typer.echo(f"Fingertip bbox saved to {fingertip_bbox_path}")

    typer.echo("Done!")


@app.command(help="Run fingertip detection using a custom YOLOv8n model.")
def detect(
    source_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    model_file: Path = typer.Argument(..., help="Path to the custom YOLOv8n model weights."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
):
    # Create output directories
    images_dir = Path(target_directory) / "fingertips"
    labels_dir = Path(target_directory) / "bbox"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO model
    model = YOLO(str(model_file))

    # Run inference on all images in the source directory
    results = model(source_directory, stream=True, max_det=1, save_txt=True, save_crop=True)

    # Consume generator to process all images
    for _ in results:
        pass

    runs_dir = Path(settings["runs_dir"])
    results_images_dir = runs_dir / "detect" / "predict" / "crops" / "Fingertip"
    results_labels_dir = runs_dir / "detect" / "predict" / "labels"

    # Move cropped images and labels to the output directory
    for file in Path(results_images_dir).glob("*.jpg"):
        typer.echo(f"Moving {file} to {images_dir}")
        shutil.move(str(file), str(images_dir / file.name))

    for file in Path(results_labels_dir).glob("*.txt"):
        typer.echo(f"Moving {file} to {labels_dir}")
        shutil.move(str(file), str(labels_dir / file.name))

    # Clean up
    typer.echo(f"Cleaning up {runs_dir}")
    shutil.rmtree(runs_dir)

    typer.echo("Done!")


@app.command(help="Generate binary mask through background subtraction.")
def subtract(
    source_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
):
    # Create output directories
    masks_dir = Path(target_directory) / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Create a new session using U2NET model
    rembg_session = new_session("u2net")

    for image_path in tqdm(list(Path(source_directory).glob("*.jpg"))):
        # Read RGB sample image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Subtract background from the image
        mask = remove(image, only_mask=True, session=rembg_session)

        # Post-process mask
        mask = post_process_mask(mask)

        # Save fingertip mask
        mask_path = masks_dir / image_path.with_suffix(".png").name
        cv2.imwrite(str(mask_path), mask)
        typer.echo(f"Fingertip mask saved to {mask_path}")

    typer.echo("Done!")

@app.command(help="Generate a fingertip image quality assessment report.")
def score(
    source_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    mask_directory: Path = typer.Argument(..., help="Path to the fingertip masks directory."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
    report_file: Path = typer.Option("quality_scores.csv", help="Path to the output report file."),
):
    # Create output directories
    report_dir = Path(target_directory)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file_path = report_dir / report_file

    # Open the CSV file
    with open(report_file_path, "w", newline="") as csvfile:
        fieldnames = ["Image name", "Sharpness", "Contrast", "Binary Mask Coverage"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Process each image in the directory
        image_paths = list(Path(source_directory).glob("*.jpg"))
        for image_path in tqdm(image_paths):
            mask_path = Path(mask_directory) / image_path.with_suffix(".png").name

            # Read the image and mask
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # Compute quality scores
            sharpness, contrast, coverage = quality_scores(image, mask)

            typer.echo(
                f"Image: {image_path.name}, Sharpness: {sharpness:.2f}, Contrast: {contrast:.2f}, Binary Mask Coverage: {coverage:.2f}"
            )

            # Write the data to the CSV file
            writer.writerow(
                {
                    "Image name": image_path.name,
                    "Sharpness": sharpness,
                    "Contrast": contrast,
                    "Binary Mask Coverage": coverage,
                }
            )

    typer.echo(f"Quality scores saved to {report_file_path}")

@app.command(help="Run fingertip enhancement (bilateral filter and CLAHE).")
def enhance(
    source_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    mask_directory: Path = typer.Argument(..., help="Path to the fingertip masks directory."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
    area: float = typer.Option(65.0, help="Binary Mask Coverage percentage threshold."),
):
    # Create output directories
    images_dir = Path(target_directory) / "enhancement"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(source_directory).glob("*.jpg"))
    for image_path in tqdm(image_paths):
        mask_path = Path(mask_directory) / image_path.with_suffix(".png").name

        # Read fingertip and mask images (Grayscale)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        _, _, coverage = quality_scores(image, mask)

        if coverage < area:
            typer.echo(f"Skipping {image_path} due to low ({coverage:.2f}) coverage percentage.")
            continue

        typer.echo(
            f"Threshold: {area}; Image: {image_path}, Binary Mask Coverage: {coverage:.2f}"
        )

        fingertip = fingertip_enhancement(image)

        # Save enhanced fingertip without background
        fingertip = cv2.bitwise_and(fingertip, fingertip, mask=mask)
        fingertip_path = images_dir / image_path.name
        cv2.imwrite(str(fingertip_path), fingertip)
        typer.echo(f"Enhanced fingertip image saved to {fingertip_path}")

    typer.echo("Done!")

@app.command(help="Run mean adaptive thresholding.")
def binarize(
    source_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
    block_size: int = typer.Option(11, help="Block size."),
):
    binarized_dir = Path(target_directory) / "binarized"
    binarized_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(list(Path(source_directory).glob("*.jpg"))):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        binarized = fingertip_thresholding(image, block_size)

        # Save binarized fingertip
        binarized_path = binarized_dir / image_path.with_suffix(".png").name
        cv2.imwrite(str(binarized_path), binarized)
        typer.echo(f"Binarized fingertip image saved to {binarized_path}")

    typer.echo("Done!")

@app.command(help="Transform fingertip images into fingerprint images.")
def convert(
    source_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
):
    fingerprint_dir = Path(target_directory) / "mapping"
    fingerprint_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(list(Path(source_directory).glob("*.png"))):
        fingertip = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        fingerprint = fingerprint_mapping(fingertip)
        fingerprint = fingerprint_enhancement(fingerprint)

        # Save enhanced fingerprint
        fingerprint_path = fingerprint_dir / image_path.with_suffix(".png").name
        cv2.imwrite(str(fingerprint_path), fingerprint)
        typer.echo(f"Contactless to contact mapping saved to {fingerprint_path}")

    typer.echo("Done!")

