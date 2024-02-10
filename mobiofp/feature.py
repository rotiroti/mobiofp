import typer
import cv2
import pickle
import imutils

from tqdm import tqdm
from pathlib import Path

app = typer.Typer()


@app.command(help="Extract features using OpenCV ORB.")
def extract(
    mapping_directory: Path = typer.Argument(
        ..., help="Path to the input mapping directory."
    ),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
    thinning: bool = typer.Option(False, help="Whether to thin the fingerprint image."),
):
    # Create output directories
    features_dir = Path(target_directory) / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Process each image in the directory
    for image_path in tqdm(list(Path(mapping_directory).glob("*.png"))):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if thinning:
            image = imutils.skeletonize(image)

        # Instantiate ORB detector
        orb = cv2.ORB_create()

        # Find the keypoints and descriptors with ORB
        keypoints, descriptors = orb.detectAndCompute(image, None)

        # Convert keypoints to a list of tuples
        keypoints_list = [
            (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            for kp in keypoints
        ]

        # Save keypoints and descriptors
        features_path = features_dir / image_path.with_suffix(".pickle").name

        with open(features_path, "wb") as f:
            pickle.dump((keypoints_list, descriptors), f)

    typer.echo("Done!")
