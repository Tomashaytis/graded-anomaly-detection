"""
Apply VideoLoader.reflect_pad_and_resize to a single image.

Useful for presentations: shows exactly what to_tensor does to a frame.
"""

import cv2
import argparse

from pathlib import Path

from src.dataset import VideoLoader


INPUT_PATH = "../../docs/Рисунки/office_group3.png"
OUTPUT_PATH = "../../docs/Рисунки/office_group3_224.png"
SIZE = 224
CROP_BLACK = True
RATIO_THRESHOLD = 10
VERTICAL_BLUR = True


def process_image(
    input_path: str,
    output_path: str,
    size: int = SIZE,
    crop_black: bool = CROP_BLACK,
    ratio_threshold: float = RATIO_THRESHOLD,
    vertical_blur: bool = VERTICAL_BLUR,
) -> None:
    """
    Read an image, apply the same transform as VideoLoader.to_tensor,
    and save the result as a size x size PNG.

    :param input_path: path to the input image
    :param output_path: path to the output image
    :param size: target side length (size x size)
    :param crop_black: crop near-black borders before resizing
    :param ratio_threshold: ratio threshold for tiling vs cropping
    :param vertical_blur: use blur-pad instead of center-crop for wide frames
    """
    frame = cv2.imread(input_path)
    if frame is None:
        raise FileNotFoundError(f"Failed to read image: {input_path}")

    # Same pipeline as VideoLoader.to_tensor: BGR uint8 -> processed BGR uint8
    processed = VideoLoader.reflect_pad_and_resize(
        frame,
        (size, size),
        crop_black=crop_black,
        ratio_threshold=ratio_threshold,
        vertical_blur=vertical_blur,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, processed)
    print(f"Saved {output_path}  ({processed.shape[1]}x{processed.shape[0]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default=INPUT_PATH)
    parser.add_argument("output", nargs="?", default=OUTPUT_PATH)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--no-crop-black", action="store_true")
    parser.add_argument("--ratio-threshold", type=float, default=RATIO_THRESHOLD)
    parser.add_argument("--no-blur", action="store_true")
    args = parser.parse_args()

    process_image(
        input_path=args.input,
        output_path=args.output,
        size=args.size,
        crop_black=not args.no_crop_black,
        ratio_threshold=args.ratio_threshold,
        vertical_blur=not args.no_blur,
    )