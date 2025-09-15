import numpy as np
import cv2

from src.io.experiment import ExpIO
from src.calibration.roi import ROISelector

class AffineTransform:
    def __init__(self):
        self.affine_matrix = None

    def find(self, camera_image, sim_image):
        camera_selector = ROISelector(camera_image, window_name="Camera Image")
        simulation_selector = ROISelector(sim_image, window_name="Simulation Image")

        ROISelector.print_instructions()

        while True:
            camera_selector.update()
            simulation_selector.update()

            if camera_selector.check_esc_key(): # Press 'Esc' to exit selection
                camera_selector.close()
                simulation_selector.close()
                break

        if len(camera_selector.rois) < 3 or len(simulation_selector.rois) < 3:
            print("Not enough points selected. Please restart and select at least 3.")
            exit()

        camera_centers = np.array([roi.com for roi in camera_selector.rois], dtype=np.float32)
        simulation_centers = np.array([roi.com for roi in simulation_selector.rois], dtype=np.float32)

        self.affine_matrix, _ = cv2.estimateAffinePartial2D(camera_centers, simulation_centers)

    def apply(self, image, final_shape):
        final_shape_T = (final_shape[1], final_shape[0])
        aligned_camera_image = cv2.warpAffine(image, self.affine_matrix, final_shape_T)
        return aligned_camera_image

    def save(self, exp_io: ExpIO, name="affine_transform"):
        exp_io.save_npy(self.affine_matrix, name)

    def load(self, path):
        self.affine_matrix = np.load(path)

def add_ref_point(image, y, x, ref_pt_radius=4):
    """
    Add a reference point to the image at the specified coordinates (y, x).
    Args:
        image (np.ndarray): The input image where the reference point will be added.
        y (int): The y-coordinate of the reference point.
        x (int): The x-coordinate of the reference point.
        ref_pt_radius (int): The radius of the reference point to be added.
    Returns:
        np.ndarray: The image with the reference point added.
    """
    # Add a reference point at (y, x) with a radius of ref_pt_radius
    y_min = max(0, y - ref_pt_radius)
    y_max = min(image.shape[0], y + ref_pt_radius)
    x_min = max(0, x - ref_pt_radius)
    x_max = min(image.shape[1], x + ref_pt_radius)

    image[y_min:y_max, x_min:x_max] = 1

    return image


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.cgh import patterns
    from src.cgh.propagator import PropagatorNumpy

    slm_shape = (50, 50)
    period = 5

    checkerboard = patterns.checkerboard(slm_shape, period)
    phase = checkerboard * np.pi

    prop = PropagatorNumpy(slm_shape, far_upscale=4, far_centred=False)

    near_field = np.exp(1j * phase)
    far_field = prop.get_far_field(near_field)
    far_field_intensity = prop.get_intensity(far_field)

    affine_transform = AffineTransform()
    affine_transform.find(far_field_intensity, far_field_intensity)

    aligned_image = affine_transform.apply(far_field_intensity, far_field_intensity.shape)
    plt.imshow(aligned_image, cmap='gray')
    plt.title("Aligned Image")
    plt.colorbar()
    plt.show()
