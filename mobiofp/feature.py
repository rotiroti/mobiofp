import pickle
from pathlib import Path

import cv2
import imutils
import pandas as pd
import typer
from tqdm import tqdm

app = typer.Typer()


# @app.command(help="Extract features using OpenCV ORB.")
# def extract(
#     mapping_directory: Path = typer.Argument(..., help="Path to the input mapping directory."),
#     target_directory: Path = typer.Argument(..., help="Path to the output directory."),
#     method: str = typer.Option("orb", help="Feature extraction method."),
#     thinning: bool = typer.Option(False, help="Whether to thin the fingerprint image."),
#     features: int = typer.Option(500, help="Maximum number of features to retain."),
# ):
#     # Create output directory
#     features_dir = Path(target_directory) / "features"
#     features_dir.mkdir(parents=True, exist_ok=True)

#     # Process each image in the directory
#     for image_path in tqdm(list(Path(mapping_directory).glob("*.png"))):
#         image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

#         if thinning:
#             image = imutils.skeletonize(image)

#         descriptors = None
#         if method == "orb":
#             descriptor = cv2.ORB_create(nfeatures=features)
#         elif method == "sift":
#             descriptor = cv2.SIFT_create(nfeatures=features)
#         elif method == "brief":
#             descriptor = cv2.BRIEF_create(nfeatures=features)
#         elif method == "brisk":
#             descriptor = cv2.BRISK_create(nfeatures=features)
#         elif method == "freak":
#             descriptor = cv2.FastFeatureDetector_create(nfeatures=features)
#         else:
#             raise ValueError(f"Unknown method: {method}")

#         # Find the keypoints and descriptors
#         keypoints, descriptors = descriptor.detectAndCompute(image, None)

#         # Convert keypoints to a list of tuples
#         keypoints_list = [
#             (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
#             for kp in keypoints
#         ]

#         # Save keypoints and descriptors
#         features_path = features_dir / image_path.with_suffix(".pickle").name

#         with open(features_path, "wb") as f:
#             pickle.dump((keypoints_list, descriptors), f)

#     typer.echo("Done!")


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
