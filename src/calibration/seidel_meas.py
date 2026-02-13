import cv2
import numpy as np
import dearpygui.dearpygui as dpg
import time
from src.io.experiment import ExpIO
from src.hardware.camera_thor.camera import CameraThorlabs
from src.hardware.slm_santec.slm import SLMSantec
from src.cgh.propagator import AberrCoefficients, SeidelPropagatorPatches
from src.cgh.optimisers import PhaseOptimBGD
import os
from scipy.spatial import distance
from src.cgh.propagator import PropagatorTorch
from src.cgh.patterns import make_cross_square_tiled, make_cross_square
from skopt import gp_minimize

# Constants
EXPOSURE_TIME = 2e-2  # Default exposure time (s)
WAVELENGTH = 633      # nm
NUM_ITERATIONS = 100
NUM_ITERATIONS_FULL = 200

NUM_COEFFS = 6  # Number of Seidel coefficients
FAR_SCALE = 1

COEFF_RANGE = 39
COEFF_STEP_SIZE = 3 


class SeidelPhase:
    """
    Represents a Seidel aberration phase generator.
    This class computes the phase term based on the provided coefficients.
    """
    def __init__(self, slm_shape: tuple, num_tiles: int):
        """
        Initialize with the SLM shape.
        :param slm_shape: Tuple (height, width) of the SLM.
        """
        self.P, self.Q = slm_shape  # SLM shape
        self.M, self.N = 2 * self.P, 2 * self.Q  # Propagation grid size

        self.Y, self.X = self._get_xy_grid()  # Normalized coordinates for SLM
        self.V, self.U = self._get_uv_grid()  # Normalized coordinates for

         # Get normalized coordinates for Seidel aberrations. They are defined so that the radius <= 1.0. 
        r = np.sqrt(self.P**2 + self.Q**2)
        self.Xn = self.X * 2.0 / r
        self.Yn = self.Y * 2.0 / r

        un = self.U[0] * np.sqrt(2.0) # Take the first row of U
        vn = self.V[:,0] * np.sqrt(2.0) # Take the first column of V

        self.num_tiles = num_tiles
        tile_dims = np.array([self.M // num_tiles, self.N // num_tiles])

        self.u0 = np.zeros((num_tiles, num_tiles))
        self.v0 = np.zeros((num_tiles, num_tiles))

        for tile_r in range(self.num_tiles):
            for tile_s in range(self.num_tiles):
                # Center of the tile in normalized coordinates
                self.v0[tile_r, tile_s] = vn[tile_r * tile_dims[0] + tile_dims[0] // 2]
                self.u0[tile_r, tile_s] = un[tile_s * tile_dims[1] + tile_dims[1] // 2]

    def _get_xy_grid(self):
        p_indices = np.fft.fftshift(np.fft.fftfreq(self.P, d=1/self.P))
        q_indices = np.fft.fftshift(np.fft.fftfreq(self.Q, d=1/self.Q))
        Y, X = np.meshgrid(p_indices, q_indices, indexing='ij')
        return Y, X

    def _get_uv_grid(self):
        m_indices = np.fft.fftshift(np.fft.fftfreq(self.M))
        n_indices = np.fft.fftshift(np.fft.fftfreq(self.N))
        V, U = np.meshgrid(m_indices, n_indices, indexing='ij')
        return V, U

    def get_phase(self, aberr: AberrCoefficients, tile_y: int, tile_x: int) -> np.ndarray:
        v0 = self.v0[tile_y, tile_x]
        u0 = self.u0[tile_y, tile_x]

        # For now only calculate the wavefront curvature and tilt.
        beta = np.arctan2(v0, u0)
        h2 = u0**2 + v0**2
        h = np.sqrt(h2)  # Normalized radius
        R2 = self.Xn**2 + self.Yn**2  # Normalized radius squared
        Xr = self.Xn * np.cos(beta) + self.Yn * np.sin(beta)

        phase = np.zeros_like(R2)

        if aberr.coma != 0:
            phase += aberr.coma * h * R2 * Xr
        if aberr.astigmatism != 0:
            phase += aberr.astigmatism * h2 * Xr**2
        if aberr.field_curvature != 0:
            phase += aberr.field_curvature * h2 * R2
        if aberr.distortion != 0:
            phase += aberr.distortion * h2 * h * Xr
        if aberr.tilt_x != 0:
            phase += aberr.tilt_x * u0 * R2
        if aberr.tilt_y != 0:
            phase += aberr.tilt_y * v0 * R2

        return phase

class ImageViewer:
    """
    Handles image display in a DearPyGui drawlist, with pan & zoom.
    """

    def __init__(self):
        # Zoom & pan state
        self.zoom = 0.5                # Default zoom
        self.offset = [0, 0]           # Pan offset in pixels
        self.dragging = False
        self.last_mouse = (0, 0)
        self.base_width = 1            # Will be set when first image arrives
        self.base_height = 1

        # Tags for DPG items
        self.texture_tag = "image_texture"
        self.drawlist_tag = "image_drawlist"
        self.draw_image_tag = "draw_image"

    def create(self):
        """
        Create the texture registry, the window & drawlist, and register handlers.
        Must be called after dpg.create_context().
        """

        # Initialize a dummy image to get initial dimensions
        height, width = CameraThorlabs.get_shape()  # Get camera shape for initial texture size
        img = np.zeros((height, width))
        w, h, img_data = self.process_image(img)

        # 1) Dynamic texture placeholder
        with dpg.texture_registry(show=False):
            # 1x1 dummy RGB pixel initially
            dpg.add_dynamic_texture(width=w, height=h, default_value=img_data, tag=self.texture_tag)

        # 2) Create window + drawlist for image display
        with dpg.window(label="Image Viewer", no_close=True, tag="image_viewer_window"):
            # drawlist: fixed size initially; you can adjust as needed
            with dpg.drawlist(width=w, height=h, tag=self.drawlist_tag):
                # Initial dummy draw_image: pmin==pmax means not visible
                dpg.draw_image(self.texture_tag,
                               pmin=(0, 0),
                               pmax=(0, 0),
                               uv_min=(0, 0),
                               uv_max=(1, 1),
                               tag=self.draw_image_tag)

        # 3) Register mouse handlers in a handler registry
        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=self.handle_drag)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=self.stop_drag)
            dpg.add_mouse_wheel_handler(callback=self.handle_zoom)

    def process_image(self, image: np.ndarray) -> tuple:
        """
        Process the input image to get dimensions and normalized data.
        - image: input image array (can be grayscale or RGB)
        - Returns: (width, height, normalized_image_data)
        """
        img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        h, w, _ = img.shape
        img_data = img.flatten().astype(np.float32) / 255.0

        return w, h, img_data

    def update_image(self, image: np.ndarray):
        """
        Called by the main App when a new image arrives.
        - image_data: flattened float32 array in [0,1], RGBA, length = width*height*4
        - width, height: original image dimensions
        This sets the texture, updates base dimensions, centers the image in the drawlist, then redraws.
        """
        # Process the image to get dimensions and data
        width, height, image_data = self.process_image(image)

        # Update base dims
        self.base_width = width
        self.base_height = height

        # Update texture
        dpg.set_value(self.texture_tag, image_data)

        # Redraw
        self.redraw()

    def redraw(self):
        """
        Update the drawn image coordinates in the drawlist according to current zoom & offset.
        """
        draw_w = int(self.base_width * self.zoom)
        draw_h = int(self.base_height * self.zoom)
        pmin = (self.offset[0], self.offset[1])
        pmax = (self.offset[0] + draw_w, self.offset[1] + draw_h)

        # Update the draw_image item
        dpg.configure_item(self.draw_image_tag,
                           pmin=pmin,
                           pmax=pmax,
                           uv_min=(0, 0),
                           uv_max=(1, 1))

    def handle_drag(self, sender, app_data):
        """
        Pan image when left button dragging within the drawlist.
        app_data: (delta_x, delta_y) or similar; but we use get_mouse_pos.
        """
        # Only pan if mouse is over the drawlist
        if not dpg.is_item_hovered(self.drawlist_tag):
            return

        mouse = dpg.get_mouse_pos(local=False)
        if not self.dragging:
            # Start dragging
            self.dragging = True
            self.last_mouse = mouse
        else:
            # Continue dragging
            dx = mouse[0] - self.last_mouse[0]
            dy = mouse[1] - self.last_mouse[1]
            self.offset[0] += dx
            self.offset[1] += dy
            self.last_mouse = mouse
            self.redraw()

    def stop_drag(self, sender, app_data):
        """
        End pan mode on mouse release.
        """
        self.dragging = False

    def handle_zoom(self, sender, app_data):
        """
        Zoom in/out centered on cursor when mouse wheel scroll, only if hovering drawlist.
        app_data: positive for scroll up, negative for scroll down.
        """
        if not dpg.is_item_hovered(self.drawlist_tag):
            return

        # Get global mouse position
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        # Get drawlist top-left in global coords
        rect_min = dpg.get_item_rect_min(self.drawlist_tag)
        # Compute mouse position relative to image origin before offset:
        rel_x = mouse_x - rect_min[0] - self.offset[0]
        rel_y = mouse_y - rect_min[1] - self.offset[1]

        old_zoom = self.zoom
        # Zoom factor per wheel notch
        factor = 1.1 if app_data > 0 else 0.9
        # Clamp zoom
        self.zoom = max(0.1, min(5.0, self.zoom * factor))
        new_zoom = self.zoom
        scale = new_zoom / old_zoom

        # Adjust offset so that the point under cursor remains under cursor after zoom
        self.offset[0] -= rel_x * (scale - 1)
        self.offset[1] -= rel_y * (scale - 1)

        self.redraw()

class SeidelMeasurementApp:
    """
    Class for measuring Seidel aberrations across the field of view by applying known aberrations on the SLM and analyzing the captured images. It provides methods for finding the affine transform, applying Seidel aberrations, capturing images, and analyzing them to extract the aberration coefficients. Two methods for finding the coefficients are provided: a grid search and a Gaussian process optimization. The class handles camera/SLM, hologram setup, and communicates with ImageViewer.
    """

    def __init__(self, exp_time_affine, tile_indices, first_last_tile_y=(3,21), coeff_range = COEFF_RANGE, coeff_step_size = COEFF_STEP_SIZE, exposure_time=EXPOSURE_TIME, field_inc=None):
        self.camera = None
        self.slm = None
        self.cam_exposure_time = exposure_time
        self.defocus_coeff = 0.0

        self.slm_shape = None

        self.last_image = None  # Last captured image for display
        self.exp = None  # Experiment IO object for saving data

        self.target_name_list = ["Points", "Cross", "Markers"]
        self.target_name = "Points"  # Default target type

        # Store the exposure time for affine transform capture
        self.exp_time_affine = exp_time_affine

        # Prepare hologram
        self.setup_holo(field_inc=field_inc)

        # Initialize Seidel coefficients and generator
        self.seidel_coeffs = np.zeros(NUM_COEFFS)  # Initialize Seidel coefficients
        self.aberr = AberrCoefficients(
            coma = 0,
            astigmatism = 0,
            field_curvature = 0,
            distortion = 0,
            tilt_x = 0,
            tilt_y = 0
        )
        self.coeff_names = ["Coma", "Astigmatism", "Field Curvature", "Distortion", "Tilt X", "Tilt Y"]
        self.seidel_gen = SeidelPhase(self.slm_shape, num_tiles=self.num_tiles)

        # Intitialize tile selection
        self.tile_x = 0
        self.tile_y = 0

        # Store tile indices for analysis
        self.tile_indices = tile_indices

        # Typically the camera can see more tiles along x than y, but the affine transform transforms the image to a square region.
        # Therefore, we need to limit the tile selection sliders to the smaller number of tiles along y.
        self.first_tile_y, self.last_tile_y = first_last_tile_y

        # Range and step size for the coefficient scan
        self.coeff_range = coeff_range
        self.coeff_step_size = coeff_step_size

        # Status tags
        self.save_status_tag = "save_status_text"
        self.analyze_image_button_tag = "analyze_image_button"
        self.tune_seidel_coeffs_button_tag = "tune_seidel_coeffs_button"
        self.tune_seidel_coeffs_g_button_tag = "tune_seidel_coeffs_g_button"

        # Set a flag to indicate whether to correct Seidel aberrations on the full field or just the selected tile. 
        self.full_field_optim = False

        # Initialize DearPyGui context, camera and SLM
        self.run()

    def setup_holo(self, field_inc=None):
        """
        Prepare wavefront correction hologram if needed.
        """
        self.field_inc = field_inc

        self.slm_shape = SLMSantec.get_shape()

        def make_target(target_shape, num_markers=20):
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
            marker = np.zeros(shape, dtype=np.float32)
            marker[shape[0] // 2, shape[1] // 2] = 1  # Center pixel

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
        
        # 1. Initialize the propagator with the incident field
        prop = PropagatorTorch(self.slm_shape, field_inc=self.field_inc, scale=FAR_SCALE, square_far_field=True)

        # 2. Define the target using the provided function
        self.num_tiles = np.gcd(*self.slm_shape) // 10
        self.tile_shape = (prop.M // self.num_tiles, prop.N // self.num_tiles)

        # Spots over the whole far field
        self.target_points = make_target((prop.M, prop.N), num_markers=self.num_tiles)

        # Cross markers
        self.target_markers = make_cross_square_tiled((prop.M, prop.N), num_markers=self.num_tiles, marker_size=15 * FAR_SCALE)
        self.marker_single = make_cross_square(self.tile_shape, size=15 * FAR_SCALE)

        # Centred cross lines
        self.target_lines_center = np.zeros_like(self.target_points, dtype=np.float32)  # Initialize target as zeros
        h, w = self.target_lines_center.shape
        self.target_lines_center[h//2, :] = 1
        self.target_lines_center[:, w//2] = 1

        # 3. Create the Optimiser
        optim = PhaseOptimBGD(prop, target_int=self.target_points, num_iter=NUM_ITERATIONS, batch_size=1, precalc_near_field=True)

        # 4. Run the optimization for each pattern
        phase_init = np.random.rand(*self.slm_shape) * 2 * np.pi  # Random initial phase
        self.phase_points = optim.optim(phase_init, learning_rate=0.2)
        optim.set_target(self.target_lines_center)
        self.phase_cross = optim.optim(phase_init, learning_rate=0.2)
        optim.set_target(self.target_markers)
        self.phase_markers = optim.optim(phase_init, learning_rate=0.2)

    def setup_gui(self):
        """
        Create control window, then launch viewport & start DearPyGui.
        """
        with dpg.window(label="Controls", no_close=True, tag="controls_window"):
            with dpg.theme(tag="save_button_theme"):
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 80, 80, 255))       # Reddish background
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (220, 100, 100, 255))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (180, 60, 60, 255))
                    dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 240, 240, 255))       # Off-white text

            with dpg.theme(tag="analyze_button_theme"):
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (80, 200, 80, 255))       # Green background
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (100, 220, 100, 255))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (60, 180, 60, 255))
                    dpg.add_theme_color(dpg.mvThemeCol_Text, (240, 255, 240, 255))       # Off-white text

            dpg.add_text("Camera controls:")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Capture", callback=self.capture_image_button)
                save_btn = dpg.add_button(label="Save", callback=self.save_image)
                dpg.bind_item_theme(save_btn, "save_button_theme")
                dpg.add_text("", tag=self.save_status_tag, show=False)

            dpg.add_text("Analyzis controls:")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Find Affine", callback=self.get_affine_transform)
                dpg.add_button(label="Tune Coeffs (Sweep)", callback=self.tune_seidel_coeffs, tag=self.tune_seidel_coeffs_button_tag, show=False)
                dpg.bind_item_theme(dpg.last_item(), "analyze_button_theme")
                dpg.add_button(label="Tune Coeffs (Gauss)", callback=self.tune_seidel_coeffs_gauss, tag=self.tune_seidel_coeffs_g_button_tag, show=False)
                dpg.bind_item_theme(dpg.last_item(), "analyze_button_theme")
                dpg.add_button(label="Analyze and Save", callback=self.analyze_image, tag=self.analyze_image_button_tag, show=False)
                dpg.bind_item_theme(dpg.last_item(), "save_button_theme")

            dpg.add_text("Target controls:")
            # Add radio buttons for target selection
            dpg.add_radio_button(items=self.target_name_list, default_value=self.target_name_list[0],
                                 callback=lambda s, d: setattr(self, 'target_name', d),
                                 tag="target_selection_radio", horizontal=True)

            # Full field optimization toggle
            dpg.add_text("Optimization controls:")
            dpg.add_checkbox(label="Full Field Optimization", default_value=False,
                             callback=lambda s, d: setattr(self, 'full_field_optim', d),
                             tag="full_field_optimization_toggle")
                
            # Exposure Time slider+input
            self.add_linked_slider_input(
                label="Exposure Time",
                default_value=self.cam_exposure_time,
                min_value=1e-5,
                max_value=1e-1,
                callback=lambda s, d: setattr(self, 'cam_exposure_time', d),
                tag_prefix="exp"
            )

            # Tile selection sliders
            self.add_linked_slider_input(
                label="Tile X",
                default_value=self.tile_x,
                min_value=0,
                max_value=self.num_tiles - 1,
                callback=lambda s, d: setattr(self, 'tile_x', int(d)),
                tag_prefix="tile_x"
            )
            self.add_linked_slider_input(
                label="Tile Y",
                default_value=self.tile_y,
                min_value=0,
                max_value=self.num_tiles - 1,
                callback=lambda s, d: setattr(self, 'tile_y', int(d)),
                tag_prefix="tile_y"
            )

            dpg.add_text("Wavefront controls:")
            # Add sliders for all Seidel coefficients
            for i, name in enumerate(self.coeff_names):
                def create_callback(index):
                    def callback(s, d):
                        self.seidel_coeffs[index] = d
                    return callback

                self.add_linked_slider_input(
                    label=f"{name} Coefficient",
                    default_value=self.seidel_coeffs[i],
                    min_value=-COEFF_RANGE,
                    max_value=COEFF_RANGE,
                    callback=create_callback(i),
                    tag_prefix=f"seidel_{i}"
                )

        # Create and show viewport
        dpg.create_viewport(title='Hologram Image Display', width=1200, height=900)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        # Start the DearPyGui event loop
        dpg.start_dearpygui()
        # Cleanup
        dpg.destroy_context()

    def add_linked_slider_input(self, label, default_value, min_value, max_value, callback,
                                format="%.5f", width=200, parent=None, tag_prefix=""):
        """
        Adds a linked slider+input field. Disables +/- buttons on input via step=0.
        """
        slider_tag = f"{tag_prefix}_slider"
        input_tag = f"{tag_prefix}_input"

        def on_slider_change(sender, app_data):
            dpg.set_value(input_tag, app_data)
            callback(sender, app_data)

        def on_input_change(sender, app_data):
            dpg.set_value(slider_tag, app_data)
            callback(sender, app_data)

        group_args = {"horizontal": True}
        if parent:
            group_args["parent"] = parent

        with dpg.group(**group_args):
            dpg.add_slider_double(label=label,
                                  default_value=default_value,
                                  min_value=min_value,
                                  max_value=max_value,
                                  format=format,
                                  width=width,
                                  callback=on_slider_change,
                                  tag=slider_tag)
            dpg.add_input_double(default_value=default_value,
                                 min_value=min_value,
                                 max_value=max_value,
                                 format=format,
                                 width=100,
                                 callback=on_input_change,
                                 tag=input_tag,
                                 step=0)
            
    def update_linked_slider_input(self, tag_prefix, value):
        """
        Update both slider and input field for a given tag prefix.
        """
        slider_tag = f"{tag_prefix}_slider"
        input_tag = f"{tag_prefix}_input"
        dpg.set_value(slider_tag, value)
        dpg.set_value(input_tag, value)
            
    def set_aberr_coeffs(self):
        """
        Update the aberration coefficients based on current slider values.
        This is called when tuning coefficients or analyzing images.
        """
        self.aberr.coma = self.seidel_coeffs[0]
        self.aberr.astigmatism = self.seidel_coeffs[1]
        self.aberr.field_curvature = self.seidel_coeffs[2]
        self.aberr.distortion = self.seidel_coeffs[3]
        self.aberr.tilt_x = self.seidel_coeffs[4]
        self.aberr.tilt_y = self.seidel_coeffs[5]

    def capture_image(self, phase) -> np.ndarray:
        """
        Display phase pattern on SLM, capture camera image.
        """
        grayscale = SLMSantec.phase_to_grayscale(phase)
        self.slm.display(grayscale)

        # Set exposure and capture
        self.camera.set_exposure_time(self.cam_exposure_time)
        time.sleep(0.3)
        img = self.camera.capture()
        return img

    def get_phase(self, target_in: np.ndarray, phase_in: np.ndarray) -> np.ndarray:
        # Compute combined phase
        self.set_aberr_coeffs()  # Update the aberration coefficients from sliders

        if self.full_field_optim:
            # Use full field optimization
            seidel_prop = SeidelPropagatorPatches(self.slm_shape, aberr=self.aberr, field_inc=self.field_inc, scale=FAR_SCALE, num_tiles=self.num_tiles, bound=20, use_blending=False, vectorise=True, square_far_field=True)
            optim = PhaseOptimBGD(seidel_prop, target_int=target_in, num_iter=NUM_ITERATIONS, batch_size=1, precalc_near_field=True)

            # Run the optimization
            phase_init = np.random.rand(*self.slm_shape) * 2 * np.pi  # Random initial phase
            phase = optim.optim(phase_init, learning_rate=0.2)
        else:
            phase_defocus = self.seidel_gen.get_phase(self.aberr, self.tile_y, self.tile_x)
            phase = phase_in + phase_defocus

        return phase
    
    def get_affine_transform(self, sender=None, app_data=None):
        def analyze_image_center(img: np.ndarray) -> np.ndarray:
            def fit_cross_lines(img: np.ndarray, mask: np.ndarray = None):
                img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                if mask is not None:
                    img_norm = cv2.bitwise_and(img_norm, img_norm, mask=mask)

                edges = cv2.Canny(img_norm, 50, 150, apertureSize=3)

                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                        minLineLength=100, maxLineGap=20)

                horiz_lines = []
                vert_lines = []

                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    if abs(angle) < 10:
                        horiz_lines.append((x1, y1, x2, y2))
                    elif abs(angle - 90) < 10 or abs(angle + 90) < 10:
                        vert_lines.append((x1, y1, x2, y2))

                h_points = np.array(
                    [(x1, y1) for (x1, y1, x2, y2) in horiz_lines] + [(x2, y2) for (x1, y1, x2, y2) in horiz_lines],
                    dtype=np.float32
                )
                v_points = np.array(
                    [(x1, y1) for (x1, y1, x2, y2) in vert_lines] + [(x2, y2) for (x1, y1, x2, y2) in vert_lines],
                    dtype=np.float32
                )

                vx1, vy1, x1, y1 = cv2.fitLine(h_points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                vx2, vy2, x2, y2 = cv2.fitLine(v_points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

                A = np.array([[vx1, -vx2], [vy1, -vy2]])
                b = np.array([x2 - x1, y2 - y1])
                t = np.linalg.lstsq(A, b, rcond=None)[0]

                cx = int(round(float(x1 + t[0] * vx1)))
                cy = int(round(float(y1 + t[0] * vy1)))
                center = (cx, cy)

                return {
                    "center": center,
                    "vx1": vx1, "vy1": vy1, "x1": x1, "y1": y1,
                    "vx2": vx2, "vy2": vy2, "x2": x2, "y2": y2,
                }
        
            # Step 1: Initial center estimate
            result1 = fit_cross_lines(img)
            center1 = result1["center"]

            # Step 2: Create mask to ignore bright center
            mask = np.ones_like(img, dtype=np.uint8) * 255
            cv2.circle(mask, center1, radius=100, color=0, thickness=-1)  # mask radius = 30px

            # Step 3: Second pass without bright center
            result2 = fit_cross_lines(img, mask=mask)
            center2 = result2["center"]

            # Step 4: Visualize result on original (unmasked) image
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_color = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

            height, width = img.shape[:2]

            # Draw fitted horizontal line
            pt1_h = (int(result2["x1"] - result2["vx1"] * width), int(result2["y1"] - result2["vy1"] * width))
            pt2_h = (int(result2["x1"] + result2["vx1"] * width), int(result2["y1"] + result2["vy1"] * width))
            cv2.line(img_color, pt1_h, pt2_h, (0, 255, 0), 1)

            # Draw fitted vertical line
            pt1_v = (int(result2["x2"] - result2["vx2"] * height), int(result2["y2"] - result2["vy2"] * height))
            pt2_v = (int(result2["x2"] + result2["vx2"] * height), int(result2["y2"] + result2["vy2"] * height))
            cv2.line(img_color, pt1_v, pt2_v, (0, 255, 0), 1)

            # Draw centers
            cv2.circle(img_color, center1, radius=4, color=(255, 0, 0), thickness=-1)  # initial center = blue
            cv2.circle(img_color, center2, radius=6, color=(0, 0, 255), thickness=-1)  # refined center = red

            return img_color, center2
        
        def analyze_image_points(img: np.ndarray, center=None, radius=40, img_color=None) -> np.ndarray:
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            if center is None:
                # If no center provided, use the image center
                height, width = img_norm.shape
                center = (width // 2, height // 2)

            # Only select points within a certain radius from the center, but avoid the center itself
            mask = np.zeros(img_norm.shape, dtype=np.uint8)
            mask = cv2.circle(mask, center, radius * 4, 255, -1)
            mask = cv2.circle(mask, center, radius, 0, -1)
            img_mask = cv2.bitwise_and(img_norm, img_norm, mask=mask)

            # Otsu's thresholding to binarize the image
            _, img_thres = cv2.threshold(img_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find all blobs using connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_thres, connectivity=8)

            # Merge close centroids by averaging their positions
            min_distance = 10  # Minimum distance between centroids to consider them separate
            merged_centroids = []

            for centroid in centroids[1:]:  # Skip the background
                merged = False
                for i, existing in enumerate(merged_centroids):
                    if distance.euclidean(centroid, existing) <= min_distance:
                        # Average the positions if they are too close
                        merged_centroids[i] = (existing + centroid) / 2
                        merged = True
                        break
                if not merged:
                    merged_centroids.append(centroid)

            if img_color is None:
                img_color = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

            # Draw the merged centroids
            for centroid in merged_centroids:
                cv2.circle(img_color, (int(centroid[0]), int(centroid[1])), 5, (0, 255, 0), -1)

            return img_color, merged_centroids
    
        exp_time_init = self.cam_exposure_time
        # Lower exposure time for affine analysis
        self.update_linked_slider_input("exp", self.exp_time_affine)  # Update the exposure time slider
        self.cam_exposure_time = self.exp_time_affine  # Temporarily set a lower exposure time for analysis

        if self.exp is None:
            self.exp = ExpIO(name="wavefront_manual_obs", description="Manual wavefront correction experiment")

        # Capture images and analyze them to find the affine transform
        img = self.capture_image(self.phase_cross) # Capture the image with the cross pattern
        img_color, center = analyze_image_center(img)  # Analyze the captured image
        self.exp.save_image(img_color, "affine_cross_fit.png")  # Save the cross fit image
        img = self.capture_image(self.phase_points)  # Capture the image with the center pattern
        img_color, points_obs = analyze_image_points(img, center=center, radius=40, img_color=img_color)  # Analyze the captured image with points
        self.exp.save_image(img_color, "affine_points_fit.png")  # Save the points fit image
        _, points_target = analyze_image_points(self.target_points, radius=50)  # Analyze the target points

        self.affine_transform = cv2.estimateAffinePartial2D(np.array(points_obs), np.array(points_target), method=cv2.RANSAC)[0]

        self.viewer.update_image(img_color)

        self.update_linked_slider_input("exp", exp_time_init)  # Update the exposure time slider
        self.cam_exposure_time = exp_time_init  # Restore the original exposure time

        dpg.show_item(self.analyze_image_button_tag)  # Show the Analyze button after finding the affine transform
        dpg.show_item(self.tune_seidel_coeffs_button_tag)  # Show the Tune Coefficients button after finding the affine transform
        dpg.show_item(self.tune_seidel_coeffs_g_button_tag)  # Show the Tune Coefficients (gaussian) button after finding the affine transform
        
        # Save the affine transform and points for later use
        self.exp.save_npy(self.affine_transform, "affine_transform")

        # Compute the adjusted affine transform that captures all the tiles:
        # Adjust the affine transform to shift by half a tile shape so that all the tiles are visible and centered
        adjusted_transform = self.affine_transform.copy()
        adjusted_transform[0, 2] += self.tile_shape[1] // 2  # Shift x by half tile width
        adjusted_transform[1, 2] += self.tile_shape[0] // 2  # Shift y by half tile height

        self.adjusted_transform = adjusted_transform.copy()

        self.exp.save_npy(self.adjusted_transform, "affine_transform_adjusted")

        self.img_tiles_shape = (self.tile_shape[1] * (self.num_tiles + 1), self.tile_shape[0] * (self.num_tiles + 1))

        print("Affine transform:", self.affine_transform)
        print("Adjusted affine transform:", self.adjusted_transform)
        print("Tile shape:", self.tile_shape)
        print("Shape of all tiles combined:", self.img_tiles_shape)

    def image_to_tiles(self, img: np.ndarray, save_image=False) -> np.ndarray:
        """
        Convert the image to tiles based on the current affine transform.
        This will reshape the image into a grid of tiles for analysis.
        """
        if self.adjusted_transform is None:
            print("Affine transform not found. Please run 'Find Affine' first.")
            return None

        # Apply the adjusted affine transform to the image
        img_transformed = cv2.warpAffine(img, self.adjusted_transform, self.img_tiles_shape)

        img_transformed_col = cv2.cvtColor(cv2.normalize(img_transformed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for i in range(self.num_tiles+1):
            for j in range(self.num_tiles+1):
                # Write tile indices as text on each tile of the transformed image for visualization
                text = f"[{i},{j}]"
                position = (
                    int((j + 0.5) * self.tile_shape[1] - 20),
                    int((i + 0.5) * self.tile_shape[0] + 10)
                )
                # Write in green
                cv2.putText(img_transformed_col, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # TODO: Does not work properly with the viewer at the moment
        # self.viewer.update_image(img_transformed)  # Update the viewer with the affine transformed image

        # Remove the selected number of top and bottom rows to save on memory
        img_cropped = img_transformed[self.first_tile_y * self.tile_shape[0]:(self.last_tile_y+1) * self.tile_shape[0], :]

        # Store the image
        if save_image:
            if self.exp is None:
                self.exp = ExpIO(name="wavefront_manual_obs", description="Manual wavefront correction experiment")
            
            self.exp.save_image(img_transformed_col, "aligned_image")
            self.exp.save_image(img, "raw_image")
        
        # Reshape into tiles
        num_tiles_y = self.last_tile_y+1 - self.first_tile_y
        tiles = img_cropped.reshape(num_tiles_y, self.tile_shape[0], self.num_tiles + 1, self.tile_shape[1]).swapaxes(1, 2)
        
        return tiles

    def analyze_image(self, sender=None, app_data=None) -> float:
        tiles = self.image_to_tiles(self.last_image, save_image=True)  # Convert the last captured image to tiles
        
        tiles_selected = []

        for indices in self.tile_indices:
            tile_y, tile_x = indices
            tile = tiles[tile_y, tile_x]
            tiles_selected.append(tile)
            tile_norm = cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            if self.exp is None:
                self.exp = ExpIO(name="wavefront_manual_obs", description="Manual wavefront correction experiment")

            print(f"Saving tile ({tile_y}, {tile_x})")
            self.exp.save_image(tile_norm, f"tile_{tile_y}_{tile_x}.png")
            self.exp.save_npy(tile, f"tile_{tile_y}_{tile_x}.npy")

        analyzed_loss = self.analyze_tiles(tiles_selected)  # Analyze the tiles to compute the loss
        print(f"Analyzed Loss: {analyzed_loss:.4f}") 

    def analyze_tiles(self, tiles):
        tile_metrics = []

        for tile in tiles:
            # Normalize and threshold
            tile_norm = cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, tile_thresh = cv2.threshold(tile_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            tile_masked = cv2.bitwise_and(tile_norm, tile_norm, mask=tile_thresh)

            # MSE difference
            marker_norm = cv2.normalize(self.marker_single, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            mse = np.mean((tile_masked - marker_norm) ** 2)
            tile_metrics.append(mse)

        loss = np.mean(np.array(tile_metrics))

        return loss

    def analyze_tiles_com(self, tiles) -> float:
        """
        Find the distortion coefficient that minimizes the loss based on the current image.
        This will adjust the distortion coefficient to center the markers within the tiles.
        """
        tile_metrics = []

        for tile in tiles:
            # Normalize and threshold
            tile_norm = cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, tile_thresh = cv2.threshold(tile_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            tile_masked = cv2.bitwise_and(tile_norm, tile_norm, mask=tile_thresh)

            # Find distance of the COM from the center
            com = cv2.moments(tile_masked)
            if com["m00"] == 0:
                continue
            com_x = int(com["m10"] / com["m00"])
            com_y = int(com["m01"] / com["m00"])
            center_x = tile_masked.shape[1] // 2
            center_y = tile_masked.shape[0] // 2
            distance2 = (com_x - center_x) ** 2 + (com_y - center_y) ** 2
            tile_metrics.append(distance2)

        loss = np.mean(np.array(tile_metrics))

        return loss
    
    def tune_seidel_coeffs_gauss(self, sender=None, app_data=None):
        
        """
        Tune Seidel coefficients based on the current image and affine transform.
        This will adjust the coefficients to minimize the loss based on the analyzed image.
        """
        def get_tiles(tile_indices, save_image=False):
            # phase = self.get_phase(target_in=self.target_points)  # Get the current phase pattern
            phase = self.get_phase(target_in=self.target_markers, phase_in=self.phase_markers)
            img = self.capture_image(phase)
            self.last_image = np.copy(img)  # Store the raw image for potential future use

            tiles = self.image_to_tiles(self.last_image, save_image=save_image)  # Convert the last captured image to tiles
            tiles_selected = []

            # Select only the specified tiles
            for indices in tile_indices:
                tile_y, tile_x = indices
                tile = tiles[tile_y, tile_x]
                tiles_selected.append(tile)

            return tiles_selected

        if self.affine_transform is None:
            print("Affine transform not found. Please run 'Find Affine' first.")
            return
        
        # Ensure the Seidel optimization is used
        self.full_field_optim = True
        dpg.set_value("full_field_optimization_toggle", self.full_field_optim)

        opt_coeffs = np.zeros(NUM_COEFFS)  # Store minimum loss coefficients
        
        tiles_init = get_tiles(self.tile_indices, save_image=True)  # Get the initial tiles based on the current coefficients

        if self.exp is None:
            self.exp = ExpIO(name="wavefront_manual_obs", description="Manual wavefront correction experiment")

        # Save the initial tiles to the experiment directory
        for i, tile in enumerate(tiles_init):
            tile_norm = cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            tile_y, tile_x = self.tile_indices[i]
            self.exp.save_image(tile_norm, f"tile_init_{tile_x}_{tile_y}.png")
            self.exp.save_npy(tile, f"tile_init_{tile_x}_{tile_y}")

        time_start = time.time()

        # First optimize the distortion coefficient using the COM method
        coeff_index = 3  # Distortion coefficient index
        
        loss_dist = np.zeros(int(2 * self.coeff_range / self.coeff_step_size) + 1)

        loss_min = np.inf
        for i, coeff in enumerate(np.arange(-self.coeff_range, self.coeff_range + 1, self.coeff_step_size)):
            self.seidel_coeffs[coeff_index] = coeff
            self.set_aberr_coeffs()

            # Update the linked slider and input for the coefficient
            self.update_linked_slider_input(f"seidel_{coeff_index}", coeff)

            # Capture the image and get the tiles
            tiles_selected = get_tiles(self.tile_indices)

            loss = self.analyze_tiles_com(tiles_selected[:4])  # Analyze using COM method for faster evaluation
            loss_dist[i] = loss

            print(f"Coefficient {coeff_index} ({self.coeff_names[coeff_index]}): {coeff}, Loss: {loss:.4f}")

            if loss < loss_min:
                loss_min = loss
                opt_coeffs[coeff_index] = coeff
            
        self.seidel_coeffs[coeff_index] = opt_coeffs[coeff_index]  # Set the optimal coefficient for distortion
        self.update_linked_slider_input(f"seidel_{coeff_index}", self.seidel_coeffs[coeff_index])  # Update the slider/input for the optimal coefficient

        print(f"Optimal coefficient for {self.coeff_names[coeff_index]}: {self.seidel_coeffs[coeff_index]} with Loss: {loss_min:.4f}")

        n_calls = 80

        loss_array = np.zeros((n_calls,))  # Store loss values for each iteration of Gaussian optimization
        coeff_array = np.zeros((n_calls, NUM_COEFFS-1))  # Store loss values for each coefficient and iteration
        test_iter = 0

        # Use gaussian process to optimize the rest of the coefficients
        def objective_function(coeffs):
            nonlocal test_iter  # Allow modification of the outer variable

            self.seidel_coeffs = np.array([coeffs[0], coeffs[1], coeffs[2], self.seidel_coeffs[3], coeffs[3], coeffs[4]])

            self.set_aberr_coeffs()

            # Capture the image and get the tiles
            tiles_selected = get_tiles(self.tile_indices)  # Get the selected tiles based on the current coefficients

            loss = self.analyze_tiles(tiles_selected)

            print(f"Coefficients: {self.seidel_coeffs}, Loss: {loss:.4f}")

            # Store the coefficients and loss for this iteration
            coeff_array[test_iter, :] = coeffs
            loss_array[test_iter] = loss
            test_iter += 1

            return loss

        res = gp_minimize(
            func=objective_function,
            dimensions=[(-COEFF_RANGE, COEFF_RANGE)] * 5,
            n_calls=80,
            n_initial_points=15,
            acq_func="EI",
            noise="gaussian",
            random_state=0,
            verbose=True
        )

        time_end = time.time()

        print(f"Seidel coefficient optimization completed in {time_end - time_start:.2f} seconds")

        print("Optimal value: %.4f" % res.fun)

        # Set the optimal coefficients found
        self.seidel_coeffs = np.array([res.x[0], res.x[1], res.x[2], self.seidel_coeffs[3], res.x[3], res.x[4]])
        opt_coeffs = self.seidel_coeffs

        print("Optimal coefficients found:")
        print(opt_coeffs)
        
        print("Loss array for Seidel distortion coefficient:")
        print(loss_dist)

        print("Loss array for Gaussian optimization:")
        print(loss_array)

        print("Tested coefficients in Gaussian optimization:")
        print(coeff_array)

        # Save the loss arrays to the experiment directory
        self.exp.save_npy(loss_dist, "seidel_loss_array_distortion")
        self.exp.save_npy(loss_array, "seidel_loss_array_gaussian")
        self.exp.save_npy(coeff_array, "seidel_tested_coeffs_gaussian")
        # Save the optimal coefficients
        self.exp.save_npy(opt_coeffs, "optimal_seidel_coeffs")

        # Save the final tiles to the experiment directory
        tiles_final = get_tiles(self.tile_indices, save_image=True)  # Get the final tiles based on the optimal coefficients
        for i, tile in enumerate(tiles_final):
            tile_norm = cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            tile_y, tile_x = self.tile_indices[i]
            self.exp.save_image(tile_norm, f"tile_final_{tile_x}_{tile_y}.png")
            self.exp.save_npy(tile, f"tile_final_{tile_x}_{tile_y}.npy")

    def tune_seidel_coeffs(self, sender=None, app_data=None):
        """
        Tune Seidel coefficients based on the current image and affine transform.
        This will adjust the coefficients to minimize the loss based on the analyzed image.
        """
        def get_tiles(tile_indices, save_image=False):
            # phase = self.get_phase(target_in=self.target_points)  # Get the current phase pattern
            phase = self.get_phase(target_in=self.target_markers, phase_in=self.phase_markers)
            img = self.capture_image(phase)
            self.last_image = np.copy(img)  # Store the raw image for potential future use

            tiles = self.image_to_tiles(self.last_image, save_image=save_image)  # Convert the last captured image to tiles
            tiles_selected = []

            # Select only the specified tiles
            for indices in tile_indices:
                tile_y, tile_x = indices
                tile = tiles[tile_y, tile_x]
                tiles_selected.append(tile)

            return tiles_selected

        if self.affine_transform is None:
            print("Affine transform not found. Please run 'Find Affine' first.")
            return
        
        # Ensure the Seidel optimization is used
        self.full_field_optim = True
        dpg.set_value("full_field_optimization_toggle", self.full_field_optim)

        coeff_order = np.array([3]) # Order to tune coefficients: Distortion, Tilt X, Tilt Y, Astigmatism, Coma, Field Curvature

        loss_array = np.zeros((len(coeff_order), int(2 * self.coeff_range / self.coeff_step_size) + 1))
        opt_coeffs = np.zeros(len(coeff_order))  # Store minimum loss coefficients
        
        tiles_init = get_tiles(self.tile_indices, save_image=True)  # Get the initial tiles based on the current coefficients

        if self.exp is None:
            self.exp = ExpIO(name="wavefront_manual_obs", description="Manual wavefront correction experiment")

        # Save the initial tiles to the experiment directory
        for i, tile in enumerate(tiles_init):
            tile_norm = cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            tile_y, tile_x = self.tile_indices[i]
            self.exp.save_image(tile_norm, f"tile_init_{tile_x}_{tile_y}.png")
            self.exp.save_npy(tile, f"tile_init_{tile_x}_{tile_y}")

        time_start = time.time()

        for i, j in enumerate(coeff_order):
            loss_min = np.inf
            for coeff in np.arange(-self.coeff_range, self.coeff_range + 1, self.coeff_step_size):
                self.seidel_coeffs[j] = coeff
                self.set_aberr_coeffs()

                # Update the linked slider and input fo the coefficient
                self.update_linked_slider_input(f"seidel_{j}", coeff)

                # Capture the image and get the tiles
                tiles_selected = get_tiles(self.tile_indices)  # Get the selected tiles based on the current coefficients

                if j == 3:  # If we are tuning the distortion coefficient
                    # For distortion coefficient, we need to analyze the tiles to find the optimal value
                    loss = self.analyze_tiles_com(tiles_selected[:4])
                else:
                    # Analyze the image to get the loss
                    loss = self.analyze_tiles(tiles_selected)

                loss_array[i, int((coeff + self.coeff_range) / self.coeff_step_size)] = loss

                print(f"Coefficient {j} ({self.coeff_names[j]}): {coeff}, Loss: {loss:.4f}")

                if loss < loss_min:
                    loss_min = loss
                    opt_coeffs[i] = coeff
            
            self.seidel_coeffs[j] = opt_coeffs[i]  # Set the optimal coefficient for this index
            self.update_linked_slider_input(f"seidel_{j}", self.seidel_coeffs[j])  # Update the slider/input for the optimal coefficient

        end_time = time.time()

        print(f"Seidel coefficient optimization completed in {end_time - time_start:.2f} seconds")

        print("Optimal coefficients found:")
        print(self.seidel_coeffs)

        self.exp.save_npy(self.seidel_coeffs, "optimal_seidel_coeffs")

        # Save the final tiles to the experiment directory
        tiles_final = get_tiles(self.tile_indices, save_image=True)  # Get the final tiles based on the optimal coefficients
        for i, tile in enumerate(tiles_final):
            tile_norm = cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            tile_y, tile_x = self.tile_indices[i]
            self.exp.save_image(tile_norm, f"tile_final_{tile_x}_{tile_y}.png")
            self.exp.save_npy(tile, f"tile_final_{tile_x}_{tile_y}.npy")

        # Get the right ordering
        idx_order = np.argsort(coeff_order) # Skip the first coefficient which is distortion and was tuned at the end as well
        opt_coeffs = opt_coeffs[idx_order]
        loss_array = loss_array[idx_order, :]

        print("Loss array for Seidel coefficients:")
        print(loss_array)
        print("Optimal coefficients found:")
        print(opt_coeffs)

        # Save the loss array to the experiment directory
        self.exp.save_npy(loss_array, "seidel_loss_array")
        # Save the optimal coefficients
        self.exp.save_npy(opt_coeffs, "optimal_seidel_coeffs_opt_coeffs")

        import sys
        print("Finished tuning Seidel coefficients. Exiting program to avoid long optimization times in the full-field method.")
        exit()
        sys.exit(0)  # Exit the program after tuning coefficients to avoid long optimization times in the full-field method

    def capture_image_button(self, sender=None, app_data=None):
        """
        Callback for Capture button. Captures, processes, and sends to viewer.
        """

        # Determine the target and phase based on the selected target name
        phase_in = None
        target_in = None
        if self.target_name == "Points":
            target_in = self.target_points
            phase_in = self.phase_points
        elif self.target_name == "Cross":
            target_in = self.target_lines_center
            phase_in = self.phase_cross
        elif self.target_name == "Markers":
            target_in = self.target_markers
            phase_in = self.phase_markers

        phase = self.get_phase(target_in=target_in, phase_in=phase_in)
        img = self.capture_image(phase)

        self.last_image = np.copy(img)  # Store the raw image for potential future use
        # Send to viewer
        self.viewer.update_image(img)

    def save_image(self):
        """
        Save the last captured image to the experiment directory.
        """
        if self.last_image is not None:
            if self.exp is None:
                self.exp = ExpIO(name="wavefront_manual_obs", description="Manual wavefront correction experiment")
            file_path = self.exp.save_image(self.last_image, "capture")
            file_name = os.path.basename(file_path)
            dpg.set_value(self.save_status_tag, f"Image saved: {file_name}")
            dpg.show_item(self.save_status_tag)
            print(f"Image {file_name} saved to experiment directory.")
        else:
            print("No image captured yet.")

    def run(self):
        """
        Open camera & SLM contexts and start the GUI.
        """
        with CameraThorlabs(exposure_time=self.cam_exposure_time) as self.camera, \
             SLMSantec(use_memory_mode=False, set_wavelength=False) as self.slm:
            # Create the DPG context once
            dpg.create_context()

            # Instantiate viewer and create its widgets
            self.viewer = ImageViewer()
            self.viewer.create()

            # Then setup controls and start GUI
            self.setup_gui()