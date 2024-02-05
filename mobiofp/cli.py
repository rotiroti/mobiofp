# # from pathlib import Path

# import concurrent.futures
# import os
# from pathlib import Path

# import cv2
# import fingerprint_enhancer
# import imutils
# import numpy as np
# import rembg
# import typer
# from ultralytics import YOLO

# from mobiofp.api import crop_image, extract_roi, to_fingerprint
# from mobiofp.dataset import YOLODatasetGenerator
# from mobiofp.unet import Segment

# app = typer.Typer()


# @app.command(help="Create a dataset for training a YOLO model.")
# def create_dataset(
#     images_dir: str = typer.Argument(..., help="Name of the images directory."),
#     labels_dir: str = typer.Argument(..., help="Name of the labels directory."),
#     classes_file: str = typer.Argument(..., help="Name of the classes file."),
#     output_dir: str = typer.Argument(
#         ..., help="Path to the output directory for the generated dataset."
#     ),
#     train_ratio: float = typer.Option(
#         0.8, help="Ratio of images to be included in the training set."
#     ),
# ):
#     generator = YOLODatasetGenerator(images_dir, labels_dir, classes_file, output_dir)
#     generator.generate_dataset(train_ratio=train_ratio)
#     typer.echo("Dataset generation complete.")


# # @app.command(help="Fingerphoto recognition (U-Net semantic segmentation).")
# # def segment(
# #     src: str = typer.Argument(..., help="Path to the input image."),
# #     model_checkpoint: str = typer.Argument(
# #         ..., help="Path to the model checkpoint file."
# #     ),
# #     rotate: bool = typer.Option(True, help="Rotate the input image by 90 degrees."),
# # ):
# #     tasks = [
# #         "Reading image",
# #         "Loading model",
# #         "Extracting ROI",
# #         "Enhancing fingertip",
# #         "Fingertip binarization",
# #         "Converting to fingerprint",
# #         "Enhancing fingerprint",
# #         "Saving fingerprint"
# #     ]

# #     with typer.progressbar(tasks, label="Processing", length=100) as progress:
# #         for task in progress:
# #             typer.echo(f"Current task: {task}")
# #             if task == "Reading image":
# #                 # Read RGB sample image
# #                 image = cv2.imread(str(src))
# #                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# #                 if rotate:
# #                     image =imutils.rotate_bound(image, 90)
# #             elif task == "Loading model":
# #                 # Load a U-Net pre-trained model
# #                 model = Segment()
# #                 model.load(model_checkpoint)
# #                 mask = model.predict(image)
# #             elif task == "Extracting ROI":
# #                 # Fingertip ROI extraction
# #                 bbox = extract_roi(mask)
# #                 fingertip = crop_image(image, bbox)
# #                 fingertip_mask =crop_image(mask, bbox)
# #             elif task == "Enhancing fingertip":
# #                 # Fingertip Enhancement (Grayscale conversion, Bilateral Filter, CLAHE)
# #                 fingertip = cv2.cvtColor(fingertip, cv2.COLOR_RGB2GRAY)
# #                 fingertip = cv2.normalize(fingertip, None, 0, 255, cv2.NORM_MINMAX)
# #                 fingertip = cv2.bilateralFilter(fingertip, 7, 50, 50)
# #                 fingertip = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(fingertip)
# #             elif task == "Fingertip binarization":
# #                 # Fingertip Binarization
# #                 fingertip = cv2.bitwise_and(fingertip, fingertip, mask=fingertip_mask)
# #                 binary = cv2.adaptiveThreshold(
# #                     fingertip, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2
# #                 )
# #             elif task == "Converting to fingerprint":
# #                 # Convert fingerphoto (fingertip) to fingerprint
# #                 fingerprint = to_fingerprint(binary)
# #             elif task == "Enhancing fingerprint":
# #                 # Fingerprint Enhancement
# #                 fingerprint = fingerprint_enhancer.enhance_Fingerprint(fingerprint)
# #             elif task == "Saving fingerprint":
# #                 # Save fingerprint
# #                 PROCESSED_DIR = "./data/processed/unet"
# #                 fingerprint_filename = Path(str(src)).stem + ".png"
# #                 fingerprint_filepath = PROCESSED_DIR + "/" + fingerprint_filename

# #                 typer.echo(f"Saving fingerprint to {fingerprint_filepath}")
# #                 ret = cv2.imwrite(fingerprint_filepath, fingerprint)
# #                 assert ret, "Failed to save fingerprint."


# @app.command(help="Fingerphoto recognition (U-Net semantic segmentation).")
# def segment_directory(
#     src_dir: str = typer.Argument(
#         ..., help="Path to the directory containing the input images."
#     ),
#     model_checkpoint: str = typer.Argument(
#         ..., help="Path to the model checkpoint file."
#     ),
#     rotate: bool = typer.Option(True, help="Rotate the input image by 90 degrees."),
#     j: int = typer.Option(1, help="Number of threads to use for processing."),
# ):
#     # Ensure the number of threads is not greater than the number of cores
#     j = min(j, os.cpu_count() or 1)

#     # Get a list of all the image files in the directory
#     image_files = list(Path(src_dir).glob("*"))

#     # Create a ThreadPoolExecutor
#     with concurrent.futures.ThreadPoolExecutor(max_workers=j) as executor:
#         # Map the segment function to the image files
#         executor.map(
#             segment,
#             image_files,
#             [model_checkpoint] * len(image_files),
#             [rotate] * len(image_files),
#         )


# def segment(src: str, model_checkpoint: str, rotate: bool):
#     # Read RGB sample image
#     image = cv2.imread(str(src))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     if rotate:
#         image = imutils.rotate_bound(image, 90)

#     # Load a U-Net pre-trained model
#     model = Segment()
#     model.load(model_checkpoint)
#     mask = model.predict(image)

#     # Fingertip ROI extraction
#     bbox = extract_roi(mask)
#     fingertip = crop_image(image, bbox)
#     fingertip_mask = crop_image(mask, bbox)

#     # Fingertip Enhancement (Grayscale conversion, Bilateral Filter, CLAHE)
#     fingertip = cv2.cvtColor(fingertip, cv2.COLOR_RGB2GRAY)
#     fingertip = cv2.normalize(fingertip, None, 0, 255, cv2.NORM_MINMAX)
#     fingertip = cv2.bilateralFilter(fingertip, 7, 50, 50)
#     fingertip = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(fingertip)

#     # Fingertip Binarization
#     fingertip = cv2.bitwise_and(fingertip, fingertip, mask=fingertip_mask)
#     binary = cv2.adaptiveThreshold(
#         fingertip, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2
#     )

#     # Convert fingerphoto (fingertip) to fingerprint
#     fingerprint = to_fingerprint(binary)

#     # Fingerprint Enhancement
#     fingerprint = fingerprint_enhancer.enhance_Fingerprint(fingerprint)

#     # Save fingerprint
#     PROCESSED_DIR = "./data/processed/unet"
#     fingerprint_filename = Path(str(src)).stem + ".png"
#     fingerprint_filepath = PROCESSED_DIR + "/" + fingerprint_filename

#     print(f"Saving fingerprint to {fingerprint_filepath}")
#     _ = cv2.imwrite(fingerprint_filepath, fingerprint)

# @app.command()
# def segmentf(
#     src: str = typer.Argument(..., help="Path to the directory containing the input images."),
#     model_checkpoint: str = typer.Argument(..., help="Path to the model checkpoint file."),
#     rotate: bool = typer.Option(True, help="Rotate the input image by 90 degrees."),
# ):
#     # Read RGB sample image
#     image = cv2.imread(str(src))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = imutils.rotate_bound(image, 90)
#     # if rotate:
#     #     image = imutils.rotate_bound(image, 90)

#     # Load a U-Net pre-trained model
#     model = Segment()
#     model.load(model_checkpoint)
#     mask = model.predict(image)

#     # Fingertip ROI extraction
#     bbox = extract_roi(mask)
#     fingertip = crop_image(image, bbox)
#     fingertip_mask = crop_image(mask, bbox)

#     # Fingertip Enhancement (Grayscale conversion, Bilateral Filter, CLAHE)
#     fingertip = cv2.cvtColor(fingertip, cv2.COLOR_RGB2GRAY)
#     fingertip = cv2.normalize(fingertip, None, 0, 255, cv2.NORM_MINMAX)
#     fingertip = cv2.bilateralFilter(fingertip, 7, 50, 50)
#     fingertip = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(fingertip)

#     # Fingertip Binarization
#     fingertip = cv2.bitwise_and(fingertip, fingertip, mask=fingertip_mask)
#     binary = cv2.adaptiveThreshold(
#         fingertip, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2
#     )

#     # Convert fingerphoto (fingertip) to fingerprint
#     fingerprint = to_fingerprint(binary)

#     # Fingerprint Enhancement
#     fingerprint = fingerprint_enhancer.enhance_Fingerprint(fingerprint)

#     # Save fingerprint
#     PROCESSED_DIR = "./data/processed/unet"
#     fingerprint_filename = Path(str(src)).stem + ".png"
#     fingerprint_filepath = PROCESSED_DIR + "/" + fingerprint_filename

#     print(f"Saving fingerprint to {fingerprint_filepath}")
#     _ = cv2.imwrite(fingerprint_filepath, fingerprint)


# # @app.command(help="Fingerphoto recognition (U-Net semantic segmentation).")
# # def segment_file(
# #     src: str = typer.Argument(..., help="Path to the input image."),
# #     model_checkpoint: str = typer.Argument(
# #         ..., help="Path to the model checkpoint file."
# #     ),
# #     rotate: bool = typer.Option(True, help="Rotate the input image by 90 degrees."),
# # ):
# #     print("Reading image...")
# #     # Read RGB sample image
# #     image = cv2.imread(str(src))
# #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# #     if rotate:
# #         image =imutils.rotate_bound(image, 90)

# #     print("Loading model...")
# #     # Load a U-Net pre-trained model
# #     model = Segment()
# #     model.load(model_checkpoint)
# #     mask = model.predict(image)

# #     print("Extracting ROI...")
# #     # Fingertip ROI extraction
# #     bbox = extract_roi(mask)
# #     fingertip = crop_image(image, bbox)
# #     fingertip_mask =crop_image(mask, bbox)

# #     print("Enhancing fingertip...")
# #     # Fingertip Enhancement (Grayscale conversion, Bilateral Filter, CLAHE)
# #     fingertip = cv2.cvtColor(fingertip, cv2.COLOR_RGB2GRAY)
# #     fingertip = cv2.normalize(fingertip, None, 0, 255, cv2.NORM_MINMAX)
# #     fingertip = cv2.bilateralFilter(fingertip, 7, 50, 50)
# #     fingertip = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(fingertip)

# #     print("Fingertip binarization...")
# #     # Fingertip Binarization
# #     fingertip = cv2.bitwise_and(fingertip, fingertip, mask=fingertip_mask)
# #     binary = cv2.adaptiveThreshold(
# #         fingertip, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2
# #     )

# #     print("Converting to fingerprint...")
# #     # Convert fingerphoto (fingertip) to fingerprint
# #     fingerprint = to_fingerprint(binary)

# #     print("Enhancing fingerprint...")
# #     # Fingerprint Enhancement
# #     fingerprint = fingerprint_enhancer.enhance_Fingerprint(fingerprint)

# #     # Save fingerprint
# #     PROCESSED_DIR = "./data/processed/unet"
# #     fingerprint_filename = Path(str(src)).stem + ".png"
# #     fingerprint_filepath = PROCESSED_DIR + "/" + fingerprint_filename

# #     print(f"Saving fingerprint to {fingerprint_filepath}")
# #     _ = cv2.imwrite(fingerprint_filepath, fingerprint)

# # @app.command(help="Fingerphoto recognition (YOLOv8n object detection).")
# # def detect(
# #     src: Path = typer.Argument(..., help="Path to the input image."),
# #     model_checkpoint: Path = typer.Argument(
# #         ..., help="Path to the model checkpoint file."
# #     ),
# #     rotate: bool = typer.Option(True, help="Rotate the input image by 90 degrees."),
# # ):
# #     # Read RGB sample image
# #     image = cv2.imread(str(src))
# #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# #     if rotate:
# #         image =imutils.rotate_bound(image, 90)

# #     model = YOLO(model_checkpoint)
# #     results = model.predict(image, conf=0.85)
# #     assert len(results) > 0, "No objects detected in the image."
# #     result = results[0]

# #     # Extract bounding boxes
# #     boxes = results[0].boxes.xyxy.tolist()
# #     x1, y1, x2, y2 = boxes[0]
# #     fingertip = image[int(y1) : int(y2), int(x1) : int(x2)]

# #     # Fingertip Enhancement (Grayscale conversion, Bilateral Filter, CLAHE)
# #     fingertip = cv2.cvtColor(fingertip, cv2.COLOR_RGB2GRAY)
# #     fingertip = cv2.normalize(fingertip, None, 0, 255, cv2.NORM_MINMAX)
# #     fingertip = cv2.bilateralFilter(fingertip, 7, 50, 50)
# #     fingertip = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(fingertip)

# #     # Use Rembg to remove background
# #     fingertip_mask = rembg.remove(fingertip, only_mask=True)

# #     # Post-process fingertip mask
# #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# #     fingertip_mask = cv2.morphologyEx(fingertip_mask, cv2.MORPH_OPEN, kernel, iterations=2)
# #     fingertip_mask = cv2.GaussianBlur(
# #         fingertip_mask, (5, 5), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT
# #     )
# #     fingertip_mask = np.where(fingertip_mask < 127, 0, 255).astype(np.uint8)

# #     # Fingertip Binarization
# #     fingertip = cv2.bitwise_and(fingertip, fingertip, mask=fingertip_mask)
# #     binary = cv2.adaptiveThreshold(
# #         fingertip, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2
# #     )

# #     # Convert fingerphoto (fingertip) to fingerprint
# #     fingerprint = to_fingerprint(binary)

# #     # Fingerprint Enhancement
# #     fingerprint = fingerprint_enhancer.enhance_Fingerprint(fingerprint)

# #     # Save fingerprint
# #     PROCESSED_DIR = "./data/processed/yolo"
# #     fingerprint_filename = Path(str(src)).stem + ".png"
# #     fingerprint_filepath = PROCESSED_DIR + "/" + fingerprint_filename

# #     cv2.imwrite(fingerprint_filepath, fingerprint)


# @app.command(help="Fingerphoto recognition (YOLOv8n object detection).")
# def detect_directory(
#     src_dir: str = typer.Argument(
#         ..., help="Path to the directory containing the input images."
#     ),
#     model_checkpoint: str = typer.Argument(
#         ..., help="Path to the model checkpoint file."
#     ),
#     rotate: bool = typer.Option(True, help="Rotate the input image by 90 degrees."),
#     j: int = typer.Option(1, help="Number of threads to use for processing."),
# ):
#     # Ensure the number of threads is not greater than the number of cores
#     j = min(j, os.cpu_count() or 1)

#     # Get a list of all the image files in the directory
#     image_files = list(Path(src_dir).glob("*"))

#     # Create a ThreadPoolExecutor
#     with concurrent.futures.ThreadPoolExecutor(max_workers=j) as executor:
#         # Map the segment function to the image files
#         executor.map(
#             detect,
#             image_files,
#             [model_checkpoint] * len(image_files),
#             [rotate] * len(image_files),
#         )


# def detect(src: str, model_checkpoint: str, rotate: bool):
#     image = cv2.imread(str(src))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     if rotate:
#         image = imutils.rotate_bound(image, 90)

#     model = YOLO(model_checkpoint, )
#     results = model.predict(image, conf=0.85)
#     assert len(results) > 0, "No objects detected in the image."
#     result = results[0]

#     # Extract bounding boxes
#     boxes = results[0].boxes.xyxy.tolist()
#     x1, y1, x2, y2 = boxes[0]
#     fingertip = image[int(y1) : int(y2), int(x1) : int(x2)]

#     # Fingertip Enhancement (Grayscale conversion, Bilateral Filter, CLAHE)
#     fingertip = cv2.cvtColor(fingertip, cv2.COLOR_RGB2GRAY)
#     fingertip = cv2.normalize(fingertip, None, 0, 255, cv2.NORM_MINMAX)
#     fingertip = cv2.bilateralFilter(fingertip, 7, 50, 50)
#     fingertip = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(fingertip)

#     # Use Rembg to remove background
#     fingertip_mask = rembg.remove(fingertip, only_mask=True)

#     # Post-process fingertip mask
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     fingertip_mask = cv2.morphologyEx(
#         fingertip_mask, cv2.MORPH_OPEN, kernel, iterations=2
#     )
#     fingertip_mask = cv2.GaussianBlur(
#         fingertip_mask, (5, 5), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT
#     )
#     fingertip_mask = np.where(fingertip_mask < 127, 0, 255).astype(np.uint8)

#     # Fingertip Binarization
#     fingertip = cv2.bitwise_and(fingertip, fingertip, mask=fingertip_mask)
#     binary = cv2.adaptiveThreshold(
#         fingertip, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2
#     )

#     # Convert fingerphoto (fingertip) to fingerprint
#     fingerprint = to_fingerprint(binary)

#     # Fingerprint Enhancement
#     fingerprint = fingerprint_enhancer.enhance_Fingerprint(fingerprint)

#     # Save fingerprint
#     PROCESSED_DIR = "./data/processed/yolo"
#     fingerprint_filename = Path(str(src)).stem + ".png"
#     fingerprint_filepath = PROCESSED_DIR + "/" + fingerprint_filename

#     print(f"Saving fingerprint to {fingerprint_filepath}")
#     _ = cv2.imwrite(fingerprint_filepath, fingerprint)


# @app.command(help="Match two fingerprints.")
# def match(
#     fingerprint1: Path = typer.Argument(..., help="Path to the first fingerprint."),
#     fingerprint2: Path = typer.Argument(..., help="Path to the second fingerprint."),
#     method: str = typer.Option("feature", help="Method to use for matching."),
# ):
#     # Read the fingerprints
#     fingerprint1 = cv2.imread(str(fingerprint1), cv2.IMREAD_GRAYSCALE)
#     fingerprint2 = cv2.imread(str(fingerprint2), cv2.IMREAD_GRAYSCALE)

#     # Perform fingerprint matching
#     if method == "feature":
#         # Feature-based matching
#         sift = cv2.SIFT_create()
#         kp1, des1 = sift.detectAndCompute(fingerprint1, None)
#         kp2, des2 = sift.detectAndCompute(fingerprint2, None)

#         # FLANN parameters
#         FLANN_INDEX_KDTREE = 1
#         index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#         search_params = dict(checks=50)

#         # Matching
#         flann = cv2.FlannBasedMatcher(index_params, search_params)
#         matches = flann.knnMatch(des1, des2, k=2)

#         # Ratio test
#         good = []
#         for m, n in matches:
#             if m.distance < 0.7 * n.distance:
#                 good.append(m)

#         typer.echo(f"Number of good matches: {len(good)}")
#     # else:
#     #     # Template-based matching
#     #     res = cv2.matchTemplate(fingerprint1, fingerprint2, cv2.TM_CCOEFF_NORMED)
#     #     _, similarity, _, _ = cv2.minMaxLoc(res)
#     #     typer.echo(f"Similarity: {similarity}")


# if __name__ == "__main__":
#     app()
