import pickle
import cv2
from pathlib import Path

import typer
import random
from tqdm import tqdm

app = typer.Typer()


@app.command(help="Select a random instance for each finger and create a template.")
def select(
    features_directory: Path = typer.Argument(
        ..., help="Path to the input features directory."
    ),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
):
    subjects = {}
    illuminations = set()
    backgrounds = set()

    # 1. Read all the features and create a dictionary with the following structure:
    typer.echo("Reading features...")
    for features_path in tqdm(list(Path(features_directory).glob("*.pickle"))):
        filename = features_path.stem

        subject_id, illumination, finger_id, background, instance_id = filename.split(
            "_"
        )
        subject_id, finger_id, instance_id = (
            int(subject_id),
            int(finger_id),
            int(instance_id),
        )

        illuminations.add(illumination)
        backgrounds.add(background)
        subjects.setdefault(subject_id, {}).setdefault(finger_id, []).append(
            instance_id
        )

    # 2. Check that all features have the same illumination and background
    typer.echo("Checking features...")
    if len(illuminations) > 1 or len(backgrounds) > 1:
        raise ValueError(
            "All the images must have the same illumination and background"
        )

    # 3. Create a list of templates that will be used for the enrollment.
    output_files = []
    illumination = illuminations.pop()
    background = backgrounds.pop()

    typer.echo("Selecting templates...")
    for subject_id in subjects:
        for finger_id in subjects[subject_id]:
            # Pick a random instance for each finger
            instance_id = random.choice(subjects[subject_id][finger_id])

            feature_filename = f"{subject_id}_{illumination}_{finger_id}_{background}_{instance_id}.pickle"
            template_filename = (
                f"{subject_id}_{illumination}_{finger_id}_{background}.pickle"
            )

            output_file = {
                "feature": features_directory / feature_filename,
                "template": target_directory / template_filename,
            }
            output_files.append(output_file)

    # 4. Copy the selected templates to the target directory
    target_directory.mkdir(parents=True, exist_ok=True)

    typer.echo("Copying templates...")
    for output_file in tqdm(output_files):
        output_file["template"].write_bytes(output_file["feature"].read_bytes())

    typer.echo("Done!")


@app.command(help="Compute the similarity score between two templates.")
def score(
    probe: Path = typer.Argument(..., help="Path to the probe template."),
    gallery: Path = typer.Argument(..., help="Path to the gallery template."),
    threshold: float = typer.Option(0.7, help="Threshold for the similarity score."),
):
    # Read the features
    with open(probe, "rb") as f:
        probe_keypoints, probe_descriptors = pickle.load(f)
        probe_keypoints = [
            cv2.KeyPoint(x, y, _size, _angle, _response, _octave, _class_id)
            for x, y, _size, _angle, _response, _octave, _class_id in probe_keypoints
        ]

    with open(gallery, "rb") as f:
        gallery_keypoints, gallery_descriptors = pickle.load(f)
        gallery_keypoints = [
            cv2.KeyPoint(x, y, _size, _angle, _response, _octave, _class_id)
            for x, y, _size, _angle, _response, _octave, _class_id in gallery_keypoints
        ]

    params = {"algorithm": 1, "trees": 10}
    matches = cv2.FlannBasedMatcher(params).knnMatch(
        probe_descriptors, gallery_descriptors, k=2
    )
    match_points = []

    for m, n in matches:
        if m.distance < threshold * n.distance:
            match_points.append(m)

    keypoints = 0

    if len(probe_keypoints) < len(gallery_keypoints):
        keypoints = len(probe_keypoints)
    else:
        keypoints = len(gallery_keypoints)

    score = len(match_points) / keypoints * 100

    typer.echo(f"Probe: {probe.name}")
    typer.echo(f"Gallery: {gallery.name}")
    typer.echo(f"Similarity score: {score:.2f}")
