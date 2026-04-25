# Linear Algebra Document Scan using homography

A perspective transformation tool that uses homography and SVD to warp and straighten images based on selected points.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Command

Transform an image by straightening a perspective-distorted region:

```bash
python main.py <input_image> <output_image> <width> <height> --points '<JSON_POINTS>'
```

where `'<JSON_POINTS>'` is a list like so:
`[[x_coord_1, y_coord_1], [x_coord_2, y_coord_2], ...]`

### Arguments

- `input_image`: Path to the input image
- `output_image`: Path to save the transformed image
- `width`: Output rectangle width in pixels
- `height`: Output rectangle height in pixels
- `--points`: JSON array of `[x, y]` coordinates (minimum 4 points)

### Examples

**4-point transformation (corners only):**

```bash
python main.py image.png output.png 600 400 \
  --points '[[100, 50], [550, 75], [580, 350], [50, 320]]'
```

**8-point transformation (corners + edge midpoints):**

```bash
python main.py image.png output.png 600 400 \
  --points '[[271, 463], [296, 372], [318, 290], [237, 268], [157, 245], [129, 323], [98, 417], [178, 439]]'
```

## Latex Report

https://www.overleaf.com/read/hhshzgnkwxcv#8e5d3d

## Videos:
- Adam Rudko: https://youtu.be/OskHgZnBTJY
- Maksym Holovin: https://www.youtube.com/watch?v=Dk2WuIS3InQ
- Anton Deputat: https://youtu.be/zN5XqBQNmD8

## Testing

Run the complete automated suite:

```bash
pytest -q
```

The suite includes:

- Unit tests for homography algebra (`H * H^-1`, null-space check `A h ~= 0`)
- Pixel-level interpolation tests against hand-computed bilinear values
- Visual regression tests versus OpenCV `warpPerspective` on benchmark document images
- Robustness tests for degenerate point sets and extreme perspective cases

## Section 7 Extensions (Investigation)

Two extension paths are now prototyped in `src/extensions.py`:

- SVD confidence metric `rho = sigma_8 / sigma_9`:
  - `compute_svd_confidence_from_design_matrix(A)`
  - `compute_svd_confidence_for_points(input_points, width, height)`

- Automated corner detection using an edge-based pipeline:
  - Gaussian blur -> Canny edges -> contour approximation -> quadrilateral selection
  - `auto_detect_document_corners(image, ...)`
  - `detect_and_warp_document(image, width, height, ...)`

These are covered by tests in `tests/test_extensions.py`.
