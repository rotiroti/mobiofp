import typer
import shutil
import yaml
import cv2
import mobiofp
import fingerprint_enhancer

from pathlib import Path
from ultralytics.utils.downloads import zip_directory
# from typing_extensions import Annotated

class DatasetGenerator:
    def __init__(self, images_dir, labels_dir, classes_file, output_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.classes_file = classes_file
        self.output_dir = Path(output_dir)

    def generate_dataset(self, train_ratio=0.8):
        # Create train and val directories
        train_dir = self.output_dir / 'train'
        val_dir = self.output_dir / 'val'

        for d in [train_dir, val_dir]:
            (d / 'images').mkdir(parents=True, exist_ok=True)
            (d / 'labels').mkdir(parents=True, exist_ok=True)

        # Get list of images
        images_list = list(Path(self.images_dir).glob('*'))
        num_train_images = int(len(images_list) * train_ratio)

        print(f'Found {len(images_list)} images. Using {num_train_images} for training and {len(images_list) - num_train_images} for validation.')

        # Detect labels file extension
        label_file_ext = list(Path(self.labels_dir).glob('*'))[0].suffix

        for i, image_path in enumerate(images_list):
            image_file = image_path.name
            label_file = image_file.replace('.jpg', label_file_ext)  # Assuming labels have same name with .txt extension
            label_path = Path(self.labels_dir) / label_file

            if i < num_train_images:
                shutil.copy(image_path, train_dir / 'images' / image_file)
                shutil.copy(label_path, train_dir / 'labels' / label_file)
            else:
                shutil.copy(image_path, val_dir / 'images' / image_file)
                shutil.copy(label_path, val_dir / 'labels' / label_file)

        # Create data.yaml file
        self.create_data_yaml()

        # Zip dataset
        self.zip_dataset()

    def create_data_yaml(self):
        data = {
            'path': f'../{self.output_dir}',  # dataset root dir
            'train': 'train',  # train images (relative to 'path')
            'val': 'val',  # val images (relative to 'path')
            'names': {
                0: 'Fingertip'
            }
        }

        with open(self.output_dir / 'fingerphoto256.yaml', 'w') as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)

    def zip_dataset(self):
        zip_directory(self.output_dir)








app = typer.Typer()

@app.command()
def generate_dataset(
    images_dir: str = typer.Argument(..., help="Name of the images directory."),
    labels_dir: str = typer.Argument(..., help="Name of the labels directory."),
    classes_file: str = typer.Argument(..., help="Name of the classes file."),
    output_dir: str = typer.Argument(..., help="Path to the output directory for the generated dataset."),
    train_ratio: float = typer.Option(0.8, help="Ratio of images to be included in the training set.")
):
    generator = DatasetGenerator(images_dir, labels_dir, classes_file, output_dir)
    generator.generate_dataset(train_ratio=train_ratio)
    typer.echo("Dataset generation complete.")

@app.command()
def segment(src: Path = typer.Argument(..., help="Path to the input image."),
            model_checkpoint: Path = typer.Argument(..., help="Path to the model checkpoint file."),
            rotate: bool = typer.Option(False, help="Rotate the input image by 90 degrees.")
):
    # Load a U-Net pre-trained model
    model = mobiofp.Segment()
    model.load(model_checkpoint)
    model.info()

    # Read RGB sample image
    image = cv2.imread(str(src))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if rotate:
        image = mobiofp.rotate_image(image, 90)

    # Fingertip segmentation
    mask = model.predict(image)

    # Fingertip ROI extraction
    bbox = mobiofp.extract_roi(mask)

    typer.echo(f"Bounding box coordinates: {bbox}")

    fingertip = mobiofp.crop_image(image, bbox)
    fingertip_mask = mobiofp.crop_image(mask, bbox)

    # Fingertip Enhancement (Grayscale conversion, Bilateral Filter, CLAHE)
    fingertip = cv2.cvtColor(fingertip, cv2.COLOR_RGB2GRAY)
    fingertip = cv2.normalize(fingertip, None, 0, 255, cv2.NORM_MINMAX)
    fingertip = cv2.bilateralFilter(fingertip, 7, 50, 50)
    fingertip = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(fingertip)

    # Fingertip Binarization
    fingertip = cv2.bitwise_and(fingertip, fingertip, mask=fingertip_mask)
    binary = cv2.adaptiveThreshold(fingertip, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2)

    # Convert fingerphoto (fingertip) to fingerprint
    fingerprint = mobiofp.to_fingerprint(binary)

     # Fingerprint Enhancement
    fingerprint = fingerprint_enhancer.enhance_Fingerprint(fingerprint)

    cv2.imshow('Fingertip', fingertip)
    cv2.imshow('Fingerprint', fingerprint)
    cv2.waitKey(0)

if __name__ == "__main__":
    app()
