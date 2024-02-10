from pathlib import Path

import cv2
import imutils
import typer
from tqdm import tqdm

from .detection import UltralyticsDataset

app = typer.Typer()


@app.command(help="Rotate dataset images by a given angle (in degrees).")
def rotate(
    source_directory: Path = typer.Argument(
        ..., help="Path to the input images directory."
    ),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
    angle: float = typer.Option(
        90,
        help="Rotation angle in degrees. Positive values mean counter-clockwise rotation.",
    ),
):
    # Create output directories
    images_dir = Path(target_directory)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Process each image in the directory
    for image_path in tqdm(list(Path(source_directory).glob("*.jpg"))):
        # Read BGR sample image
        image = cv2.imread(str(image_path))

        # Apply rotation
        result = imutils.rotate_bound(image, angle)

        # Save the rotated image
        result_path = images_dir / image_path.name
        cv2.imwrite(str(result_path), result)

        typer.echo(f"Rotated image saved to {result_path}")

    typer.echo("Done!")


@app.command(help="Resize dataset images to a given width and height.")
def resize(
    source_directory: Path = typer.Argument(
        ..., help="Path to the input images directory."
    ),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
    width: int = typer.Option(400, help="Target width in pixels."),
    height: int = typer.Option(400, help="Target height in pixels."),
):
    # Create output directories
    images_dir = Path(target_directory)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Process each image in the directory
    for image_path in tqdm(list(Path(source_directory).glob("*.jpg"))):
        # Read BGR sample image
        image = cv2.imread(str(image_path))

        # Apply resizing
        result = imutils.resize(image, width=width, height=height)

        # Save the resized image
        result_path = images_dir / image_path.name
        cv2.imwrite(str(result_path), result)

        typer.echo(f"Resized image saved to {result_path}")

    typer.echo("Done!")


@app.command(help="Create dataset for YOLO object detection.")
def create(
    images_directory: Path = typer.Argument(
        ..., help="Path to the input images directory."
    ),
    labels_directory: Path = typer.Argument(
        ..., help="Path to the input labels directory."
    ),
    target_directory: Path = typer.Argument(
        ..., help="Path to the output directory for the generated dataset."
    ),
    train_ratio: float = typer.Option(
        0.8, help="Ratio of images to be included in the training set."
    ),
):
    dataset = UltralyticsDataset(images_directory, labels_directory, target_directory)
    dataset.create(train_ratio)
