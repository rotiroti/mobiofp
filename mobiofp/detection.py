import shutil
from pathlib import Path

import cv2
import numpy as np
import rembg
import yaml
from ultralytics import YOLO
from ultralytics.utils.downloads import zip_directory


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
        Create a dataset for object detection according the ultralytics format.

        This method splits the dataset into training and validation sets based on the `train_ratio` argument,
        and copies the images and labels to the corresponding directories. It also creates a YAML file for the dataset.
        If the `compress` argument is True, it zips the dataset.

        Args:
            train_ratio (float, optional): The ratio of images to use for training. Defaults to 0.8.
        """
        # Create train and val directories
        train_dir = self.output_dir / "train"
        val_dir = self.output_dir / "val"

        for d in [train_dir, val_dir]:
            (d / "images").mkdir(parents=True, exist_ok=True)
            (d / "labels").mkdir(parents=True, exist_ok=True)

        # Get list of images
        images_list = list(Path(self.images_dir).glob("*"))
        num_train_images = int(len(images_list) * train_ratio)

        print(
            f"Found {len(images_list)} images. Using {num_train_images} for training and {len(images_list) - num_train_images} for validation."
        )

        # Detect labels file extension
        label_file_ext = list(Path(self.labels_dir).glob("*"))[0].suffix

        for i, image_path in enumerate(images_list):
            image_file = image_path.name
            label_file = image_file.replace(
                ".jpg", label_file_ext
            )  # Assuming labels have same name with .txt extension
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
            "names": {0: "Fingertip"},
        }

        with open(self.output_dir / f"{self.output_dir}.yaml", "w") as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)

    def zip_dataset(self):
        """
        Zip the dataset.

        This method zips the dataset directory and saves it to the output directory.
        """
        zip_directory(self.output_dir)


# TODO: Remove this class and use directly the YOLO class from ultralytics
class Detect:
    """
    A class used to detect objects in an image using the YOLO model.

    This class provides methods to load the YOLO model from a checkpoint, predict the objects in an image,
    get the bounding box of the detected objects, extract the region of interest (ROI) from the image,
    create a mask of the ROI, and visualize the detection results.

    Attributes:
        _model (YOLO): The YOLO model used for object detection.
        _checkpoint (str): The path to the checkpoint file for the YOLO model.

    Methods:
        info(): Print the information of the YOLO model.
        predict(image: np.ndarray, safe: bool = False): Predict the objects in an image.
        bbox(result: YOLO): Get the bounding box of the detected objects.
        roi(result: YOLO, image: np.ndarray): Extract the region of interest (ROI) from the image.
        mask(result: YOLO, image: np.ndarray): Create a mask of the ROI.
        show(result: YOLO): Visualize the detection results.
    """

    _model = None
    _checkpoint = None

    def __init__(self, checkpoint: str) -> None:
        """
        Initialize the Detect class.

        This method loads the YOLO model from a checkpoint.

        Args:
            checkpoint (str): The path to the checkpoint file for the YOLO model.
        """
        self._model = YOLO(checkpoint)
        self._checkpoint = checkpoint

    def info(self):
        """
        Print the information of the YOLO model.

        This method uses the `info` method of the YOLO model to print its information.
        """
        self._model.info()

    def predict(
        self, image: np.ndarray, safe: bool = False, device: str = "cpu", conf=0.85
    ):
        """
        Predict the objects in an image.

        This method uses the YOLO model to predict the objects in an image. If the `safe` argument is True,
        it creates a new instance of the YOLO model from the checkpoint for each prediction.

        Args:
            image (np.ndarray): The input image.
            safe (bool, optional): Whether to create a new instance of the YOLO model for each prediction. Defaults to False.

        Returns:
            YOLO: The prediction result.

        Raises:
            ValueError: If no objects are detected in the image.
        """
        local_model = self._model

        if safe:
            local_model = YOLO(self._checkpoint)

        results = local_model.predict(image, device=device, conf=conf)

        if len(results) == 0:
            raise ValueError("No objects detected in the image.")

        result = results[0]

        return result

    def bbox(self, result: YOLO) -> [int, int, int, int]:
        """
        Get the bounding box of the detected objects.

        This method extracts the bounding box of the detected objects from the prediction result.

        Args:
            result (YOLO): The prediction result.

        Returns:
            list: A list of four integers representing the bounding box (x1, y1, x2, y2).
        """
        boxes = result.boxes.xyxy.tolist()
        x1, y1, x2, y2 = boxes[0]

        return int(x1), int(y1), int(x2), int(y2)

    def roi(self, result: YOLO, image: np.ndarray) -> np.ndarray:
        """
        Extract the region of interest (ROI) from the image.

        This method extracts the ROI from the image based on the bounding box of the detected objects.

        Args:
            result (YOLO): The prediction result.
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The ROI.
        """
        x1, y1, x2, y2 = self.bbox(result)

        return image[int(y1) : int(y2), int(x1) : int(x2)]

    def mask(self, result: YOLO, image: np.ndarray) -> np.ndarray:
        """
        Create a mask of the ROI.

        This method creates a mask of the ROI by removing the background and applying some post-processing steps.

        Args:
            result (YOLO): The prediction result.
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The mask of the ROI.
        """
        roi = self.roi(result, image)

        # Use Rembg to remove background
        mask = rembg.remove(roi, only_mask=True)

        # Post-process mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.GaussianBlur(
            mask, (5, 5), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT
        )
        mask = np.where(mask < 127, 0, 255).astype(np.uint8)

        return mask

    def show(self, result: YOLO) -> np.ndarray:
        """
        Visualize the detection results.

        This method uses the `plot` method of the prediction result to visualize the detection results.

        Args:
            result (YOLO): The prediction result.

        Returns:
            np.ndarray: The visualization of the detection results.
        """
        return result.plot()
