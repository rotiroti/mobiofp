import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import typer
from tqdm import tqdm

from mobiofp.iqa import estimate_noise, laplacian_sharpness, rms_contrast, gradient_magnitude

app = typer.Typer()


@app.command(help="Generate image quality assessment report.")
def report(
    source_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
    brisque_model: Path = typer.Option(
        "./models/brisque_model_live.yml", help="Path to the BRISQUE model."
    ),
    brisque_range: Path = typer.Option(
        "./models/brisque_range_live.yml", help="Path to the BRISQUE range."
    ),
    output_file: Path = typer.Option("iqa.csv", help="Path to the output report file."),
):
    report_dir = Path(target_directory)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file_path = report_dir / output_file
    brisquer = cv2.quality.QualityBRISQUE_create(str(brisque_model), str(brisque_range))

    with open(report_file_path, "w", newline="") as csvfile:
        fieldnames = ["Image name", "Laplacian", "Noise", "Contrast", "BRISQUE"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for image_path in tqdm(list(Path(source_directory).glob("*.png"))):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            laplacian = laplacian_sharpness(image)
            noise = estimate_noise(image)
            contrast = rms_contrast(image)
            brisque = brisquer.compute(image)[0]
            writer.writerow(
                {
                    "Image name": image_path.name,
                    "Laplacian": laplacian,
                    "Noise": noise,
                    "Contrast": contrast,
                    "BRISQUE": brisque,
                }
            )

    typer.echo(f"Image quality assessment report saved to {report_file_path}")
    typer.echo("Done.")


@app.command(help="Compute the gradient magnitude of images.")
def gradient(
    source_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
):
    output_dir = Path(target_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(list(Path(source_directory).glob("*.png"))):
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        output = gradient_magnitude(image)
        output_path = output_dir / p.with_suffix(".png").name
        cv2.imwrite(str(output_path), output)
        typer.echo(f"Gradient magnitude image saved to {output_path}")

    typer.echo("Done!")
