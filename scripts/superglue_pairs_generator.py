#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pairs file for evaluation")
    parser.add_argument("input_dir", type=str, help="Directory with similarity matrices")
    parser.add_argument("output_file", type=str, help="Output file with pairs")
    parser.add_argument("--file_ext", type=str, default="png", help="File extension of images")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    gallery_templates = []
    probe_templates = []

    for title in list(input_dir.glob(f"*.{args.file_ext}")):
        subject, illumination, _, background, impression = title.stem.split("_")
        subject = int(subject)
        impression = int(impression)
        item = (subject, impression, title)

        if illumination == "i" and background == "w":
            gallery_templates.append(item)
        else:
            probe_templates.append(item)

    gallery_templates.sort()
    probe_templates.sort()

    with open(Path(output_file), "w") as f:
        for i, probe in enumerate(probe_templates):
            for j, gallery in enumerate(gallery_templates):
                p_path, g_path = probe[2], gallery[2]
                f.write(f"{str(p_path.name)} {str(g_path.name)}\n")
