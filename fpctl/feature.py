import pickle
from pathlib import Path

import cv2
import pandas as pd
import typer
from tqdm import tqdm

from mobiofp.utils import imkpts

app = typer.Typer()


@app.command(help="Extract features using OpenCV ORB.")
def extract(
    mapping_directory: Path = typer.Argument(..., help="Path to the input mapping directory."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
    thinning: bool = typer.Option(False, help="Whether to thin the fingerprint image."),
    features: int = typer.Option(500, help="Maximum number of features to retain."),
    rich_keypoints: bool = typer.Option(False, help="Whether to draw rich keypoints."),
):
    # Create output directory
    features_dir = Path(target_directory) / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the ORB detector
    orb = cv2.ORB_create(nfeatures=features)

    # Process each image in the directory
    for image_path in tqdm(list(Path(mapping_directory).glob("*.png"))):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        # Apply thinning if necessary
        if thinning:
            image = cv2.ximgproc.thinning(image)

        # Find the keypoints and descriptors
        keypoints, descriptors = orb.detectAndCompute(image, None)

        flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
        if rich_keypoints:
            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

        # Draw keypoints
        keypoints_image = imkpts(image, keypoints, flags=flags)

        # Convert keypoints to a list of tuples
        keypoints_list = [
            (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            for kp in keypoints
        ]

        # Save keypoints and descriptors
        features_path = features_dir / image_path.with_suffix(".pickle").name

        with open(features_path, "wb") as f:
            pickle.dump((keypoints_list, descriptors), f)

        typer.echo(f"Saved features to {features_path}")

        # Save keypoints image
        keypoints_image_path = features_dir / image_path.with_suffix(".png").name
        cv2.imwrite(str(keypoints_image_path), keypoints_image)
        typer.echo(f"Saved keypoints image to {keypoints_image_path}")

    typer.echo("Done!")


@app.command(help="Show information about the features.")
def info(
    features_directory: Path = typer.Argument(..., help="Path to the input features directory."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
    output_file: str = typer.Option("features_list.csv", help="Name of the output file."),
):
    # Create a set to store the subjects and a dictionary to store the finger counter
    subjects = set()
    illuminations = set()
    backgrounds = set()
    finger_counter = {}

    # Create output directory
    features_dir = Path(target_directory)
    features_dir.mkdir(parents=True, exist_ok=True)

    # Process each template in the directory
    for features_path in tqdm(list(Path(features_directory).glob("*.pickle"))):
        # Parse filename
        filename = features_path.stem
        subject_id, illumination, finger_id, background, _ = filename.split("_")

        # Convert to appropriate types
        subject_id = int(subject_id)
        finger_id = int(finger_id)

        # Add subject to set
        subjects.add(subject_id)
        if subject_id not in finger_counter:
            finger_counter[subject_id] = {1: 0, 2: 0}
        finger_counter[subject_id][finger_id] += 1

        illuminations.add(illumination)
        backgrounds.add(background)

    # Create a dataframe to store the information
    data = {"Subject ID": [], "Right Index Templates": [], "Right Middle Templates": []}

    for subject_id in sorted(subjects):
        data["Subject ID"].append(subject_id)
        data["Right Index Templates"].append(finger_counter[subject_id][1])
        data["Right Middle Templates"].append(finger_counter[subject_id][2])

    typer.echo(f"Total number of subjects: {len(subjects)}")

    # Save the dataframe to a CSV file
    features_path = features_dir / output_file
    df = pd.DataFrame(data)
    df.to_csv(features_path, index=False)

    typer.echo("Done!")
