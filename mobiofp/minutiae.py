import pickle
from pathlib import Path

import cv2
import fingerprint_feature_extractor as ffe
import numpy as np
import typer
from tqdm import tqdm

app = typer.Typer()


@app.command(help="Extract minutiae features.")
def extract(
    mapping_directory: Path = typer.Argument(
        ..., help="Path to the input mapping directory."
    ),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
    threshold: int = typer.Option(10, help="Threshold for false minutiae."),
):
    # Create output directories
    minutiae_dir = Path(target_directory) / "minutiae"
    minutiae_dir.mkdir(parents=True, exist_ok=True)

    # Process each image in the directory
    for image_path in tqdm(list(Path(mapping_directory).glob("*.png"))):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        # Extract minutiae features
        terminations, bifurcations = ffe.extract_minutiae_features(image, threshold)
        minutiae = np.array(
            [
                [
                    int(minutia.locX),
                    int(minutia.locY),
                    float(minutia.Orientation[0]),
                    1 if minutia.Type == "Termination" else 3,
                ]
                for minutia in terminations + bifurcations
            ]
        )

        # Save minutiae
        minutiae_path = minutiae_dir / image_path.with_suffix(".txt").name
        np.savetxt(str(minutiae_path), minutiae, fmt=["%d", "%d", "%.2f", "%d"])

    typer.echo("Done!")
