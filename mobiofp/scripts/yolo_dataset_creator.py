import argparse

from mobiofp.dataset import YOLODatasetGenerator


def run():
    parser = argparse.ArgumentParser(description="Create a YOLO dataset.")
    parser.add_argument("images_dir", help="Name of the images directory.")
    parser.add_argument("labels_dir", help="Name of the labels directory.")
    parser.add_argument("classes_file", help="Name of the classes file.")
    parser.add_argument(
        "output_dir", help="Path to the output directory for the generated dataset."
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of images to be included in the training set.",
    )

    args = parser.parse_args()
    generator = YOLODatasetGenerator(
        args.images_dir, args.labels_dir, args.classes_file, args.output_dir
    )
    generator.generate_dataset(train_ratio=args.train_ratio)


if __name__ == "__main__":
    run()
