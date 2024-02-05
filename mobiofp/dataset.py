import shutil
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import Sequence
from ultralytics.utils.downloads import zip_directory


class YOLODatasetGenerator:
    def __init__(self, images_dir, labels_dir, classes_file, output_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.classes_file = classes_file
        self.output_dir = Path(output_dir)

    def generate_dataset(self, train_ratio=0.8):
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

        # Zip dataset
        self.zip_dataset()

    def create_data_yaml(self):
        data = {
            "path": f"../{self.output_dir}",  # dataset root dir
            "train": "train",  # train images (relative to 'path')
            "val": "val",  # val images (relative to 'path')
            "names": {0: "Fingertip"},
        }

        with open(self.output_dir / "fingerphoto256.yaml", "w") as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)

    def zip_dataset(self):
        zip_directory(self.output_dir)


class UNETDataGenerator(Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        images,
        image_dir,
        labels,
        label_dir,
        augmentation=None,
        preprocessing=None,
        batch_size=8,
        dim=(256, 256, 3),
        shuffle=True,
    ):
        "Initialization"
        self.dim = dim
        self.images = images
        self.image_dir = image_dir
        self.labels = labels
        self.label_dir = label_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"
        batch_imgs = list()
        batch_labels = list()

        # Generate data
        for i in list_IDs_temp:
            # Store sample
            img = load_img(self.image_dir + "/" + self.images[i], target_size=self.dim)
            img = img_to_array(img) / 255.0

            # Store class
            label = load_img(
                self.label_dir + "/" + self.labels[i], target_size=self.dim
            )
            label = img_to_array(label)[:, :, 0]
            label = label != 0
            label = ndimage.binary_erosion(ndimage.binary_erosion(label))
            label = ndimage.binary_dilation(ndimage.binary_dilation(label))
            label = np.expand_dims((label) * 1, axis=2)

            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=img, mask=label)
                img, label = sample["image"], sample["mask"]

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=img, mask=label)
                img, label = sample["image"], sample["mask"]

            batch_imgs.append(img)
            batch_labels.append(label)

        return np.array(batch_imgs, dtype=np.float32), np.array(
            batch_labels, dtype=np.float32
        )
