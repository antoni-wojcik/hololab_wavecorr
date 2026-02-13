import numpy as np

# Target Pattern Generation Functions

def make_cross_square(shape, size):
    """
    Create a cross with a surrounding square as a target intensity pattern.

    Parameters:
        shape (tuple): Shape of the output array (height, width).
        size (int): Half-length of the cross arms and half-side of the square.

    Returns:
        np.ndarray: Target intensity pattern.
    """
    int_target = np.zeros(shape)

    center_y, center_x = shape[0] // 2, shape[1] // 2

    size_x = size
    size_y = int(size * shape[0] / shape[1])  # Adjust size_y to maintain aspect ratio

    # Vertical arm
    int_target[center_y - size_y:center_y + size_y + 1, center_x] = 1

    # Horizontal arm
    int_target[center_y, center_x - size_x:center_x + size_x + 1] = 1

    # Surrounding square (outline)
    int_target[center_y - size_y:center_y + size_y + 1, center_x - size_x] = 1
    int_target[center_y - size_y:center_y + size_y + 1, center_x + size_x] = 1
    int_target[center_y - size_y, center_x - size_x:center_x + size_x + 1] = 1
    int_target[center_y + size_y, center_x - size_x:center_x + size_x + 1] = 1

    return int_target

def make_cross_square_tiled(target_shape, num_markers=20, marker_size=10):
    """
    Create a target pattern for the SLM.
    The target is a cross square marker pattern that is tiled to fill the SLM shape.
    The first marker is centered at (0,0) in the SLM coordinate system.
    Args:
        slm_shape (tuple): Shape of the SLM in pixels (height, width).
        num_markers (int): Number of markers in the target pattern.
    Returns:
        target (numpy.ndarray): The target pattern for the SLM.
    """

    # Create a cross square marker pattern for the SLM
    shape = (target_shape[0] // num_markers, target_shape[1] // num_markers)
    marker = make_cross_square(shape, marker_size)

    # Tile the small target to fill the slm_shape
    repeats_y = target_shape[0] // shape[0] + 1
    repeats_x = target_shape[1] // shape[1] + 1
    target_tiled = np.tile(marker, (repeats_y, repeats_x))
    target = target_tiled[:target_shape[0], :target_shape[1]]

    # Roll so that the first tile is centered at (0,0)
    center_y, center_x = shape[0] // 2, shape[1] // 2
    target = np.roll(target, -center_y, axis=0)
    target = np.roll(target, -center_x, axis=1)

    return target
