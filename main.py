#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.homograph import homography, sort_points


def main():
    parser = argparse.ArgumentParser(
        description="Apply perspective transformation to an image using homography.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("output_image", help="Path to output image")
    parser.add_argument("width", type=int, help="Output width in pixels")
    parser.add_argument("height", type=int, help="Output height in pixels")
    parser.add_argument(
        "--points",
        type=str,
        required=True,
        help="JSON array of [x, y] coordinates. Use at least 4 points.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_image)
    if not input_path.exists():
        print(f"Error: Input image '{args.input_image}' not found", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output_image)

    try:
        points_data = json.loads(args.points)
        input_points = np.array(points_data, dtype=np.float64)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON for points: {e}", file=sys.stderr)
        sys.exit(1)

    if input_points.shape[0] < 4:
        print(
            f"Error: At least 4 points required, got {input_points.shape[0]}",
            file=sys.stderr,
        )
        sys.exit(1)

    if input_points.shape[1] != 2:
        print(
            f"Error: Each point must have 2 coordinates [x, y], got shape {input_points.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        img = np.array(Image.open(input_path))
    except Exception as e:
        print(f"Error: Could not load image: {e}", file=sys.stderr)
        sys.exit(1)

    input_points = sort_points(input_points)

    try:
        result_img = homography(img, input_points, args.width, args.height)
    except Exception as e:
        print(f"Error: Transformation failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        Image.fromarray(result_img).save(output_path)
        print(f"Success: Saved transformed image to '{output_path}'")
    except Exception as e:
        print(f"Error: Could not save image: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
