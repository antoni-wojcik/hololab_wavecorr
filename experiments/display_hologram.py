# Script to display a spoke pattern hologram using different wavefront correction methods/measurements, and save the results

from src.hardware.slm_santec.slm import SLMSantec
from src.hardware.camera_thor.camera import CameraThorlabs
from src.io.experiment import ExpIO
from src.io.santec_io import get_backplane_correction_grayscale
from src.cgh.utils import phase_to_grayscale

import numpy as np
from skimage.transform import resize
from tqdm import tqdm


# Setup
wavelength = 658 # nm Laser source wavelength
cam_exposure_time = 15e-5 # s Camera exposure time
cam_gain = 0 # dB Camera gain
slm_timeout = 5 # s SLM timeout
num_iterations = 300 # Number of iterations in the 
num_tests = 8 # Total number of experiments to perform (details below)

# Location and aspect ratio of the spoke pattern
spokes_centre_offset  = 0.18
spokes_squash_ratio = 0.934

P, Q = SLMSantec.get_shape() # Shape of the SLM in pixels
measured_wav_phase_path = r"data\experiments\wavefront_measurement\phase_wavefront.npy" # Path to the measured wavefront phase
measured_wav_amp_path = r"data\experiments\wavefront_measurement\amp_wavefront.npy" # Path to the measured wavefront amplitude
given_phase_path = r"data\backplane\backplane_phase.csv" # Path to the provided phase correction data by Santec

# -----------------------------------------------------
# Function definitions
# -----------------------------------------------------

# Calculate the spoke pattern hologram using the modified Gerchberg-Saxton algorithm
def get_holo(target_int, incident_field, force_phase_runtime=True):
    # Target amplitude
    target_amp = np.sqrt(target_int)

    # get initial phase and amplitude correction (from the square pixels)
    u,v = np.meshgrid(np.linspace(-0.5,0.5,2 * Q), np.linspace(-0.5,0.5,2 * Q)) 
    phase_init = np.exp(1j * 2 * np.pi * (u**2 + v**2))
    sinc_amp = np.sinc(u) * np.sinc(v)

    # Initialize the complex field in the image plane
    target_amp_corr = target_amp / sinc_amp
    E = target_amp_corr * phase_init
    A = np.zeros(incident_field.shape, dtype=np.complex128)

    for i in tqdm(range(num_iterations), desc="Gerchberg-Saxton Iterations"):
        # Propagate from replay field to hologram aperture, and crop the centre
        A_pad = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E)))
        A = A_pad[Q - P//2:Q + P//2, Q//2:3*Q//2]
                   
        # Apply phase-only constraint for hologram aperture
        if force_phase_runtime:
            A = np.exp(1.j*np.angle(A)) * incident_field
        else:
            A = np.exp(1.j*np.angle(A)) * np.abs(incident_field)

        # Pad with zeros to ensure that no aliasing occurs
        A_pad = np.zeros((2*Q, 2*Q), dtype=np.complex128)
        A_pad[Q - P//2:Q + P//2, Q//2:3*Q//2] = A
        
        # Propagate from hologram aperture to replay field
        E = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(A_pad)))

        # Apply the target amplitude constraint at the replay field
        E = target_amp_corr * np.exp(1.j * np.angle(E))

    # Get the final phase
    phase = np.angle(A)

    if not force_phase_runtime:
        # Use the incident field to correct the phase
        phase = np.angle(np.exp(1j * phase) * incident_field)

    grayscale_values = phase_to_grayscale(phase, grayscale_range=SLMSantec.get_grayscale_range())

    return grayscale_values

def _spoke_circle(N, spokes, ref_pt_radius=4):
    # Create grid of (y, x) coordinates
    y, x = np.indices((N, N))
    cx = cy = (N - 1) / 2.0  # center coordinates
    dx = x - cx
    dy = y - cy
    r = N / 2.0  # radius of the inscribed circle
 
    # Distance mask for the circle
    dist = np.sqrt(dx**2 + dy**2)
    mask = dist <= r
 
    # Compute angles and convert to clockwise from positive x-axis
    angles = np.arctan2(dy, dx)
    angles_cw = np.mod(-angles, 2 * np.pi)
 
    # Determine sector index for each pixel
    sector_width = np.pi / spokes
    sector_index = np.floor(angles_cw / sector_width).astype(int)
 
    # Build the binary image: 1 for white, 0 for black
    image = np.zeros((N, N), dtype=int)
    image[mask] = sector_index[mask] % 2

    image_with_points = np.pad(image, ((N//2, N//2), (N//2, N//2)), mode='constant', constant_values=1e-5)
    loc_pixel = (N, N)

    # Add reference points for the affine transformation
    for i in range(3):
        for j in range(3):
            y = i - 1
            x = j - 1
            if x == 0 or y == 0:
                if x == 0 and y == 0:
                    continue
                if x == 0:
                    y *= np.sqrt(2)
                if y == 0:
                    x *= np.sqrt(2)

                image_with_points = _add_ref_point(image_with_points, loc_pixel[0] + y * N // 2, loc_pixel[1] + x * N // 2, 2 * ref_pt_radius)
 
    return image_with_points

def _spoke_circle_stretched(N, spokes, xy_ratio):
    # Generate the spoke circle pattern at 2x the resolution
    high_res_image = _spoke_circle(2 * N, spokes, ref_pt_radius=4)
 
    # Rescale the image to the desired size
    new_size = int(N * xy_ratio) * 2
    resampled_image = np.array(resize(high_res_image.get(), (2 * N, new_size), anti_aliasing=True))
    resampled_image = resampled_image / np.max(resampled_image)

    if new_size > 2 * N:
        square_image = resampled_image[:, (new_size - 2 * N) // 2 : (new_size + 2 * N) // 2]
    else: 
        square_image = np.pad(resampled_image, ((0, 0), ((2 * N - new_size)//2, (2 * N - new_size)//2)), mode='constant', constant_values=1e-5)

    # Create a binary image
    bin_image = square_image > 0.5
    bin_image = np.array(bin_image).astype(int)
    
    return bin_image

# Add reference points to the image
def _add_ref_point(image, y, x, ref_pt_radius=4):
    # Add a reference point at (y, x) with a radius of ref_pt_radius
    y_min = max(0, y - ref_pt_radius)
    y_max = min(image.shape[0], y + ref_pt_radius)
    x_min = max(0, x - ref_pt_radius)
    x_max = min(image.shape[1], x + ref_pt_radius)

    image[y_min:y_max, x_min:x_max] = 1

    return image

# Create the spoke pattern given the aspect ratio and location
def _get_spoke_pattern(N, radius, spokes, location, xy_ratio):
    # location is in the (u,v) coordinates from [-1, -1] to [1, 1]
    spokes = _spoke_circle_stretched(2 * radius, spokes, xy_ratio=xy_ratio)

    target_int = np.ones((N, N)) * 1e-5
    loc_pixel = (location + 1) * (N // 2)
    loc_pixel = loc_pixel.astype(int)

    target_int[loc_pixel[0] - 2 * radius:loc_pixel[0] + 2 * radius, loc_pixel[1] - 2 * radius:loc_pixel[1] + 2 * radius] = spokes

    return target_int



# -----------------------------------------------------
# The experiment
# -----------------------------------------------------

# Generate the targer spoke pattern at the given position, and resized at the given ratio
target_int = _get_spoke_pattern(3840, 300, 30, np.array([spokes_centre_offset , -spokes_centre_offset ]), xy_ratio=spokes_squash_ratio)

# Load the phase and amplitude data from the measured wavefront
measured_wav_phase = np.load(measured_wav_phase_path)
measured_wav_amp = np.load(measured_wav_amp_path)
given_wav_phase = get_backplane_correction_grayscale(wavelength=wavelength, path=given_phase_path)
given_wav_phase = (np.array(given_wav_phase, dtype=np.float64) / SLMSantec.get_grayscale_range()) * 2 * np.pi

# Get the complex incident field
incident_field = measured_wav_amp * np.exp(1j * measured_wav_phase)
incident_field_given = np.exp(1j * given_wav_phase)

# Generate 8 holograms - descriptions of each given below
grayscale_values = [None] * num_tests
grayscale_values[0] = get_holo(target_int, np.ones_like(incident_field))
grayscale_values[1] = get_holo(target_int, np.abs(incident_field))
grayscale_values[2] = get_holo(target_int, incident_field / np.abs(incident_field))
grayscale_values[3] = get_holo(target_int, incident_field)
grayscale_values[4] = get_holo(target_int, incident_field_given)
grayscale_values[5] = get_holo(target_int, incident_field / np.abs(incident_field), force_phase_runtime=False)
grayscale_values[6] = get_holo(target_int, incident_field, force_phase_runtime=False)
grayscale_values[7] = get_holo(target_int, incident_field_given, force_phase_runtime=False)

# Start logging the experiment
exp = ExpIO(name="wavefront_correction", 
            description="""Testing the wavefront correction with a spoke pattern at f=6cm
0 - uncorrected
1 - corrected with amplitude
2 - corrected with phase
3 - corrected with both
4 - corrected using the given wavefront by Santec
5 - corrected with phase; phase correction only at the last step
6 - corrected with both; phase correction only at the last step
7 - corrected using the given wavefront by Santec; phase correction only at the last step
""",)

exp.save_image(target_int.get(), "target_proj")
exp.save_npy(target_int.get(), "target_proj")

target_int = _get_spoke_pattern(3840, 300, 30, np.array([spokes_centre_offset , -spokes_centre_offset ]), xy_ratio=1.0)
exp.save_image(target_int.get(), "target")
exp.save_npy(target_int.get(), "target")

# Open the SLM and camera using context managers, and save the images
with SLMSantec(wavelength=wavelength, use_memory_mode=False, set_wavelength=False, timeout=slm_timeout) as slm, \
    CameraThorlabs(exposure_time=cam_exposure_time, gain=cam_gain) as camera:

    for i in range(num_tests):
        grayscale_values_corr = grayscale_values[i].get()
        print(f"Displaying hologram {i}")

        slm.display(grayscale_values_corr)
        capture = camera.capture()

        # Save the results
        exp.save_npy(capture, f"capture_{i}")
        exp.save_npy(grayscale_values_corr, f"holo_{i}")
        exp.save_image(capture, f"capture_{i}")
        exp.save_hologram(grayscale_values_corr, f"holo_{i}")
        exp.save_image(grayscale_values_corr, f"holo_{i}")

        print(f"Saved results for hologram {i}")