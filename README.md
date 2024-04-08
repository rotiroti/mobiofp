# Mobile Biometrics Fingerphoto (MoBioFP)

## Requirements

The requirements are listed in the `requirements.txt` file. Here's a summary:

- At least **3GB** of free disk space for project dependencies!
- [Python 3.9 or later](https://www.python.org/)
- [OpenCV](https://opencv.org/)
- [Ultralytics](https://www.ultralytics.com/)
- [Tensorflow](https://www.tensorflow.org/)
- [Albumentations](https://albumentations.ai/)
- [Rembg](https://github.com/danielgatis/rembg)
- [Typer](https://typer.tiangolo.com/)

## Installation

To install the `mobiofp` package and the `fpctl` CLI application, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Create a new virtual environment to isolate project dependencies and avoid conflicting versions with system-wide installations. You can use tools like `venv`:
```
$ python -m venv new_environment
$ source ./new_environment/bin/activate
```
 or `conda`
```
$ conda create --name new_environment
$ conda activate new_environment
```
4. Install the required dependencies:

```
$ pip install .
```

or if running on **Apple M1 and M2** chipsets

```
$ pip install .
$ pip install tensorflow-metal
```
5. Verify the installation by running the `fpctl` CLI application:

```
$ fpctl --help

Usage: fpctl [OPTIONS] COMMAND [ARGS]...                                                                                                                                                            
                                                                                                                                                                                                     
 Fingerphoto Recognition Command Line Interface.                                                                                                                                                     
                                                                                                                                                                                                     
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                                                                                           │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                                                                    │
│ --help                        Show this message and exit.                                                                                                                                         │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ dataset                                                             Dataset Commands                                                                                                              │
│ fingerprint                                                         Fingerprint Commands                                                                                                          │
│ fingertip                                                           Fingertip Commands                                                                                                            │
│ quality                                                             Image Quality Commands                                                                                                        │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```


## Datasets

| Name | Description | # Samples | File Size | File Name | Location |
|---|---|---|---|---|---|
|ISPFDv1 | This consolidated dataset leverages fingerphoto images (WI, WO, NI and NO) from the IIITD SmartPhone Fingerphoto Database v1, excluding Livescan images and corrupted data from subjects 4 and 37. | 3968  | 6.96GB |  ISPFDv1.zip  | [**Download Link**](https://drive.google.com/file/d/1qKkNyO9zXkWkkCNbbXsfWuUS5PQuuTTW) |
|ISPFDv1 (90 degrees) | This consolidated dataset leverages fingerphoto images (WI, WO, NI and NO) from the IIITD SmartPhone Fingerphoto Database v1, excluding Livescan images and corrupted data from subjects 4 and 37.  All images have been preprocessed for consistency by applying a 90-degree rotation to ensure consistent finger orientation.| 3968  |6.28GB | ISPFDv1_90deg.zip  | [**Download Link**](https://drive.google.com/file/d/1df8Jaqg-5pVdgkUice5JUF4426Uj6woN) |
| Fingertip (Semantic Segmentation) | This dataset offers 297 fingerphoto images selected from the four environments (WI, WO, NI and NO) of the original IIITD SmartPhone Fingerphoto Database v1 dataset. The labels (masks) were generated using the CVAT.ai tool and assisted by the Segment Anything Model (SAM) for improved accuracy. The dataset was used to train a semantic segmentation model (U-Net) for segmenting the fingertip region in various conditions. | 297 | 529MB| fingertip297seg.zip | [**Download Link**](https://drive.google.com/file/d/1JNkH6j3QyE_FOqF88NX4ZlJgXy1nYKCC) |
| Fingertip (Object Detection) | This dataset offers 256 fingerphoto images selected from the four environments (WI, WO, NI, and NO) of the original IIITD SmartPhone Fingerphoto Database v1 dataset. The labels (bounding box) were generated using the LabelStudio tool. The dataset is arranged according to the Ultralytics [dataset](https://docs.ultralytics.com/datasets/#steps-to-contribute-a-new-dataset) folder structure, including 204 samples (80%) in the training folder and 52 (20%) in the validation folder. The dataset was used to train the object detection model (YOLOv8n) for detecting the fingertip region in various conditions. | 256 | 427MB | fingertip256obj.zip | [**Download Link**](https://drive.google.com/file/d/15akG23eTbT2TZv78kJHRImCWeKS1XL2X) |

## Jupyter Notebooks

| Name | Requirements | File Name |
|---|---|---|
| Fingertip Training Process (Object Detection) | fingertip256obj.zip | [00_rl_object_detection_training.ipynb](./notebooks/00_rl_object_detection_training.ipynb) |
| Fingertip Training Process (Semantic Segmentation) | fingertip297seg.zip  | [00_rl_semantic_segmentation_training.ipynb](./notebooks/00_rl_semantic_segmentation_training.ipynb) |
| Fingerphoto Recognition (with Fingertip Object Detection)  | fingertip-obj-[amd64\|arm64].pt  | [01_rl_fingerphoto_recognition_obj.ipynb](./notebooks/01_rl_fingerphoto_recognition_obj.ipynb)|
| Fingerphoto Recognition (with Fingertip Semantic Segmentation) | fingertip-seg-[amd64\|arm64].h5 | [01_rl_fingerphoto_recognition_seg.ipynb](./notebooks/01_rl_fingerphoto_recognition_seg.ipynb)|
| Fingertip Image Quality Assessment  | fingertips (images, masks), iqa.csv (*) | [02_rl_iqa_fingertip_enhancement.ipynb](./notebooks/02_rl_iqa_fingertip_enhancement.ipynb) |

(*) Refer to the "Fingerphoto Control CLI Application (fpctl)" section in this document for details on how to generate the Image Quality Score CSV file.

## Trained Models (weights)

| Model Name | Architecture | Filename | Location |
|---|---|---|---|
| Fingertip Semantic Segmentation (U-Net)  | amd64 | fingertip-seg-amd64.h5 | [**Download Link**](https://drive.google.com/file/d/1TLClup2s3fgkwNhXTjjVRfEQwAK18sKt) |
| Fingertip Semantic Segmentation (U-Net) | arm64 | fingertip-seg-arm64.h5  | [**Download Link**](https://drive.google.com/file/d/1lBEUzFibKANLcK1fiyfCP0ZUTCAP4_60) |
| Fingertip Object Detection (YOLOv8n)  | amd64 | fingertip-obj-amd64.pt  | [**Download Link**](https://drive.google.com/file/d/1THsT9OcTbjl_Qadw_4WqvdOEoJxVYuDW) |
| Fingertip Object Detection (YOLOv8n)  | arm64 | fingertip-obj-arm64.pt  | [**Download Link**](https://drive.google.com/file/d/1ia2Vkf4UfRI6Q_SIV1k30_WiorrlRd3K) |

## Fingerphoto Control CLI Application (fpctl)

### Dataset Commands

```
$ fpctl dataset --help
                                                                                                                                                                                                     
 Usage: fpctl dataset [OPTIONS] COMMAND [ARGS]...                                                                                                                                                    
                                                                                                                                                                                                     
 Dataset Commands                                                                                                                                                                                    
                                                                                                                                                                                                     
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                                                       │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ create                  Create dataset for YOLO object detection.                                                                                                                                 │
│ gray                    Convert dataset images to grayscale.                                                                                                                                      │
│ resize                  Resize dataset images to a given width.                                                                                                                                   │
│ rotate                  Rotate dataset images by a given angle (in degrees).                                                                                                                      │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

### Fingertip Commands

```
$ fpctl fingertip --help

                                                                                                                                                                                                    
 Fingertip Commands                                                                                                                                                                                  
                                                                                                                                                                                                     
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                                                       │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ binarize                    Run mean adaptive thresholding.                                                                                                                                       │
│ detect                      Run fingertip detection using a custom YOLOv8n model.                                                                                                                 │
│ enhance                     Run fingertip enhancement (bilateral filter and CLAHE).                                                                                                               │
│ segment                     Run fingertip segmentation using a custom U-Net model.                                                                                                                │
│ subtract                    Generate binary mask through background subtraction.                                                                                                                  │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

### Fingerprint Commands

```
$ fpctl fingerprint --help

                                                                                                                                                                                                     
 Usage: fpctl fingerprint [OPTIONS] COMMAND [ARGS]...                                                                                                                                                
                                                                                                                                                                                                     
 Fingerprint Commands                                                                                                                                                                                
                                                                                                                                                                                                     
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                                                       │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ convert                       Transform fingertip images into fingerprint images.                                                                                                                 │
│ enhance                       Run fingerprint enhancement                                                                                                                                         │
│ thinning                      Apply thinning algorithm to fingerprint images.                                                                                                                     │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Quality Commands

```
$ fpctl quality --help

                                                                                                                                                                                                     
 Usage: fpctl quality [OPTIONS] COMMAND [ARGS]...                                                                                                                                                    
                                                                                                                                                                                                     
 Image Quality Commands                                                                                                                                                                              
                                                                                                                                                                                                     
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                                                       │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ gradient                            Compute the gradient magnitude of images.                                                                                                                     │
│ report                              Generate image quality assessment report.                                                                                                                     │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Feature Extraction/Matching

Feature extraction and matching are conducted utilizing the SuperGluePretrainedNetwork repository, available at this [link](https://github.com/rotiroti/SuperGluePretrainedNetwork). For installation instructions, please consult the README.md file located in the SuperGluePretrainedNetwork submodule directory.


## TODO

- Unit Testing / Linting
- Missing docstrig for modules/functions
- Refactoring common parts for the `fpctl` CLI application
- Wrap multiple utils functions under a common class
- Docker environment for local development

## Referencese

1. On smartphone camera based fingerphoto authentication. In A. Shrestha, M. Tistarelli, M. Kirchner, K. Rathgeb, & C. Busch (Eds.), 2015 IEEE 7th International Conference on Biometrics Theory, Applications and Systems (BTAS) (pp. 1-7). IEEE. https://doi.org/10.1109/BTAS.2015.7358782

2. U-Net: Convolutional Networks for Biomedical Image Segmentation. arXiv preprint arXiv:1505.04597. https://arxiv.org/abs/1505.04597

3. Ultralytics YOLOv8. GitHub repository: https://github.com/ultralytics/ultralytics. Version 8.0.0. Authors: Glenn Jocher, Ayush Chaurasia, Jing Qiu.

4. Kauba, C.; Söllinger, D.; Kirchgasser, S.; Weissenfeld, A.; Fernández Domínguez, G.; Strobl, B.; Uhl, A. Towards Using Police Officers’ Business Smartphones for Contactless Fingerprint Acquisition and Enabling Fingerprint Comparison against Contact-Based Datasets. Sensors 2021, 21, 2248. https://doi.org/10.3390/s21072248

## License

This project is licensed under the terms of the MIT license.