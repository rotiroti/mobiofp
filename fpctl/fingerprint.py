from pathlib import Path

import cv2
import typer
from tqdm import tqdm

from mobiofp.utils import fingerprint_enhancement, fingerprint_mapping

app = typer.Typer()


@app.command(help="Transform fingertip images into fingerprint images.")
def convert(
    source_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
):
    output_dir = Path(target_directory) / "mapping"
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(list(Path(source_directory).glob("*.png"))):
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        output = fingerprint_mapping(image)
        output_path = output_dir / p.with_suffix(".png").name
        cv2.imwrite(str(output_path), output)
        typer.echo(f"Fingerprint enhanced image saved to {output_path}")
    typer.echo("Done!")


@app.command(help="Run fingerprint enhancement")
def enhance(
    source_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
):
    output_dir = Path(target_directory) / "fingerprints"
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(list(Path(source_directory).glob("*.png"))):
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        output = fingerprint_enhancement(image)
        output_path = output_dir / p.with_suffix(".png").name
        cv2.imwrite(str(output_path), output)
        typer.echo(f"Fingerprint enhanced image saved to {output_path}")
    typer.echo("Done!")


@app.command(help="Apply thinning algorithm to fingerprint images.")
def thinning(
    source_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
):
    output_dir = Path(target_directory) / "thinning"
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(list(Path(source_directory).glob("*.png"))):
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        output = cv2.ximgproc.thinning(image)
        output_path = output_dir / p.with_suffix(".png").name
        cv2.imwrite(str(output_path), output)
        typer.echo(f"Thinning image saved to {output_path}")
    typer.echo("Done!")
