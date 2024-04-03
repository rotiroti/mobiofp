import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import typer
from tqdm import tqdm
from ultralytics import YOLO, settings

from mobiofp.background import BackgroundRemoval
from mobiofp.segmentation import Segment
from mobiofp.utils import fingertip_enhancement, fingertip_thresholding

app = typer.Typer()


@app.command(help="Run fingertip segmentation using a custom U-Net model.")
def segment(
    source_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    model_file: Path = typer.Argument(..., help="Path to the custom custom U-Net model weights."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
    roi_factor: float = typer.Option(1.0, help="Region of interest factor."),
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
        bbox = segmenter.extract_roi(result, roi_factor)
        fingertip = segmenter.crop_image(image, bbox)
        fingertip_mask = segmenter.crop_image(result, bbox)

        # Normalize final image and mask
        fingertip = cv2.normalize(fingertip, None, 0, 255, cv2.NORM_MINMAX)
        fingertip = cv2.cvtColor(fingertip, cv2.COLOR_RGB2BGR)
        fingertip_mask = fingertip_mask * 255

        # Save fingertip, mask and bbox
        fingertip_path = images_dir / image_path.with_suffix(".png").name
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
        shutil.move(str(file), str(images_dir / file.with_suffix(".png").name))

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
    rembg_model: str = typer.Option("u2net", help="Rembg model to use"),
):
    output_dir = Path(target_directory) / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    remover = BackgroundRemoval(session=rembg_model)

    for p in tqdm(list(Path(source_directory).glob("*.png"))):
        image = cv2.imread(str(p))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply background removal
        output = remover.apply(image)
        output_path = output_dir / p.with_suffix(".png").name
        cv2.imwrite(str(output_path), output)
        typer.echo(f"Fingertip mask saved to {output_path}")

    typer.echo("Done!")


@app.command(help="Run fingertip enhancement (bilateral filter and CLAHE).")
def enhance(
    source_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
    diameter: int = typer.Option(
        7, help="Diameter of each pixel neighborhood that is used during filtering."
    ),
    sigma_color: int = typer.Option(100, help="Filter sigma in the color space."),
    sigma_space: int = typer.Option(100, help="Filter sigma in the coordinate space."),
    clip_limit: float = typer.Option(2.0, help="Threshold for contrast limiting."),
    tile_grid_size: tuple[int, int] = typer.Option(
        (8, 8), help="Size of grid for histogram equalization."
    ),
):
    output_dir = Path(target_directory) / "enhancement"
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(list(Path(source_directory).glob("*.png"))):
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        output = fingertip_enhancement(
            image, diameter, sigma_color, sigma_space, clip_limit, tile_grid_size
        )

        output_path = output_dir / p.with_suffix(".png").name
        cv2.imwrite(str(output_path), output)
        typer.echo(f"Enhanced fingertip image saved to {output_path}")

    typer.echo("Done!")


@app.command(help="Run mean adaptive thresholding.")
def binarize(
    source_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    mask_directory: Path = typer.Argument(..., help="Path to the fingertip masks directory."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
    block_size: int = typer.Option(19, help="Block size."),
):
    output_dir = Path(target_directory) / "binarized"
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(list(Path(source_directory).glob("*.png"))):
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        mask_path = mask_directory / p.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        output = cv2.bitwise_and(image, image, mask=mask)
        output = fingertip_thresholding(output, block_size)
        output_path = output_dir / p.name
        cv2.imwrite(str(output_path), output)
        typer.echo(f"Binarized fingertip image saved to {output_path}")

    typer.echo("Done!")
