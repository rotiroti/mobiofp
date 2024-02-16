import shutil
from pathlib import Path

import cv2
import imutils
import typer
import yaml
from tqdm import tqdm
from ultralytics.utils.downloads import zip_directory

app = typer.Typer()


def process_images(source_directory: Path, target_directory: Path, operation, operation_name: str):
    images_dir = Path(target_directory)
    images_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(list(Path(source_directory).glob("*.jpg"))):
        image = cv2.imread(str(image_path))

        # Apply operation
        result = operation(image)

        result_path = images_dir / image_path.name
        cv2.imwrite(str(result_path), result)

        typer.echo(f"{operation_name} image saved to {result_path}")

    typer.echo("Done!")


@app.command(help="Rotate dataset images by a given angle (in degrees).")
def rotate(
    source_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
    angle: float = typer.Option(
        90, help="Rotation angle in degrees. Positive values mean counter-clockwise rotation."
    ),
):
    process_images(
        source_directory, target_directory, lambda img: imutils.rotate_bound(img, angle), "Rotated"
    )


@app.command(help="Resize dataset images to a given width and height.")
def resize(
    source_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
    width: int = typer.Option(400, help="Target width in pixels."),
    height: int = typer.Option(400, help="Target height in pixels."),
):
    process_images(
        source_directory,
        target_directory,
        lambda img: imutils.resize(img, width=width, height=height),
        "Resized",
    )


@app.command(help="Convert dataset images to grayscale.")
def grayscale(
    source_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    target_directory: Path = typer.Argument(..., help="Path to the output directory."),
):
    process_images(
        source_directory,
        target_directory,
        lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        "Grayscale",
    )


@app.command(help="Create dataset for YOLO object detection.")
def create(
    images_directory: Path = typer.Argument(..., help="Path to the input images directory."),
    labels_directory: Path = typer.Argument(..., help="Path to the input labels directory."),
    target_directory: Path = typer.Argument(
        ..., help="Path to the output directory for the generated dataset."
    ),
    train_ratio: float = typer.Option(
        0.8, help="Ratio of images to be included in the training set."
    ),
):
    dataset = UltralyticsDataset(images_directory, labels_directory, target_directory)
    dataset.create(train_ratio)


class UltralyticsDataset:
    """
    A class used to prepare a dataset for object detection according the ultralytics format.

    This class provides methods to generate a dataset from a directory of images and labels,
    split the dataset into training and validation sets, create a YAML file for the dataset,
    and zip the dataset.

    Attributes:
        images_dir (str): The directory containing the images.
        labels_dir (str): The directory containing the labels.
        output_dir (Path): The directory where the output dataset will be saved.

    Methods:
        generate(train_ratio=0.8): Generate the dataset.
        create_data_yaml(): Create a YAML file for the dataset.
        zip_dataset(): Zip the dataset.
    """

    def __init__(self, images_dir, labels_dir, output_dir):
        """
        Initialize the DetectDataset class.

        Args:
            images_dir (str): The directory containing the images.
            labels_dir (str): The directory containing the labels.
            output_dir (str): The directory where the output dataset will be saved.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.output_dir = Path(output_dir)

    def create(self, train_ratio=0.8, compress=False):
        """
        Create a dataset for object detection in the ultralytics format.
        This method splits the dataset into training and validation sets
        based on the `train_ratio` argument, and copies the images and labels
        to the corresponding directories. It also creates a YAML file for the dataset.
        If the `compress` argument is True, it zips the dataset.

        Args:
            train_ratio (float, optional): The ratio of images to use for training. Defaults to 0.8.
        """
        train_dir = self.output_dir / "train"
        val_dir = self.output_dir / "val"

        for d in [train_dir, val_dir]:
            (d / "images").mkdir(parents=True, exist_ok=True)
            (d / "labels").mkdir(parents=True, exist_ok=True)

        images_list = list(Path(self.images_dir).glob("*"))
        num_train_images = int(len(images_list) * train_ratio)

        print(f"Creating dataset in {self.output_dir}...")
        print(f"{len(images_list)} images found in {self.images_dir}.")
        print(f"Creating training and validation sets with a {train_ratio} ratio.")
        print(
            f"Using {num_train_images} for training and {len(images_list) - num_train_images} for validation."
        )

        # Detect labels file extension
        label_file_ext = list(Path(self.labels_dir).glob("*"))[0].suffix

        for i, image_path in enumerate(images_list):
            image_file = image_path.name
            label_file = image_file.replace(".jpg", label_file_ext)
            label_path = Path(self.labels_dir) / label_file

            if i < num_train_images:
                shutil.copy(image_path, train_dir / "images" / image_file)
                shutil.copy(label_path, train_dir / "labels" / label_file)
            else:
                shutil.copy(image_path, val_dir / "images" / image_file)
                shutil.copy(label_path, val_dir / "labels" / label_file)

        # Create data.yaml file
        self.create_data_yaml()

        # Zip dataset (optional)
        if compress:
            self.zip_dataset()

    def create_data_yaml(self):
        """
        Create a YAML file for the dataset.

        This method creates a YAML file that specifies the paths to the training and validation images,
        and the class names.
        """
        data = {
            "path": f"../{self.output_dir}",  # dataset root dir
            "train": "train",  # train images (relative to 'path')
            "val": "val",  # val images (relative to 'path')
            "names": {0: "fingertip"},
        }

        with open(self.output_dir / f"{self.output_dir}.yaml", "w", encoding="utf-8") as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)

    def zip_dataset(self):
        """
        Zip the dataset.

        This method zips the dataset directory and saves it to the output directory.
        """
        zip_directory(self.output_dir)
