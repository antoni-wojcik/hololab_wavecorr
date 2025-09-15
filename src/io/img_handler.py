import numpy as np
import cv2
from sys import platform

# -----------------------------
# I/O Functions
# -----------------------------

# Load grayscale image from file, normalised to [0, 1]
def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)
    image = image / np.amax(image)  # Normalize to [0, 1]
    return image

# Check platform and set SVG loading method
# cairosvg causes issues on Windows, so Inkscape is used instead.
use_cairosvg = platform == "darwin"

if use_cairosvg:
    import cairosvg
    def load_svg(svg_path, size=(128, 128)):
        """
        Convert an SVG file to a grayscale 2D NumPy array normalized between 0 and 1 using OpenCV.

        Parameters:
        - svg_path: Path to the SVG file.
        - size: Tuple (width, height) to resize the output image.

        Returns:
        - 2D NumPy array with values normalized between 0 and 1.
        """
        # Convert SVG to PNG in memory
        png_data = cairosvg.svg2png(url=svg_path, output_width=size[1], output_height=size[0])

        # Convert PNG binary data to a NumPy array
        nparr = np.frombuffer(png_data, np.uint8)

        # Decode PNG image using OpenCV
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Normalize to [0, 1] range
        normalized_image = image.astype(np.float32) / 255.0

        return normalized_image
else:
    import subprocess
    import tempfile
    import os
    def load_svg(svg_path, size=(128, 128), inkscape_path=r"C:\Program Files\Inkscape\bin\inkscape.exe"):
        """
        Convert an SVG file to a grayscale 2D NumPy array normalized between 0 and 1 using OpenCV.
        Uses Inkscape (must be installed and in PATH or specify path).

        Parameters:
        - svg_path: Path to the SVG file.
        - size: Tuple (width, height) to resize the output image.
        - inkscape_path: Path to Inkscape executable if not in PATH.

        Returns:
        - 2D NumPy array with values normalized between 0 and 1.
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
            tmp_png_path = tmp_png.name

        try:
            # Run Inkscape to convert SVG to PNG
            subprocess.run([
                inkscape_path,
                svg_path,
                '--export-type=png',
                f'--export-filename={tmp_png_path}',
                f'--export-width={size[1]}',
                f'--export-height={size[0]}'
            ], check=True)

            # Load the PNG as grayscale
            image = cv2.imread(tmp_png_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError("Failed to load PNG image converted by Inkscape.")

            # Normalize
            normalized_image = image.astype(np.float32) / 255.0
            return normalized_image

        finally:
            # Clean up temporary PNG file
            if os.path.exists(tmp_png_path):
                os.remove(tmp_png_path)
    
# Save image to file, normalised to [0, 255]
def save_image(image, path):
    # Normalise the image to [0, 255]
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0

    # Convert to uint8 and save
    image = image.astype(np.uint8)

    # Get the proper path
    cv2.imwrite(path, image)