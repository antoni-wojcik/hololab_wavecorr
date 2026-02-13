from src.calibration.seidel_meas import SeidelMeasurementApp
import numpy as np

# Measurement parameters
exposure_time = 2e-2  # Default exposure time used in the measurement (s)
exposure_time_affine = 3e-4  # Set a lower exposure time for affine transform capture (s)

# Coefficient sweep parameters
coeff_range = 39 # Range of Seidel coefficients to sweep (from -coeff_range to +coeff_range)
coeff_step_size = 3 # Step size for sweeping the Seidel coefficients. 

# Only these tiles along y are considered. This has to be done because the affine transform typically adds empty areas above and below 
# the image because the camera can only see a limited vertical field of view compared to the horizontal span.
first_tile_y = 3  # First tile along y to consider
last_tile_y = 21  # Last tile along y to consider

# Select corners and center tiles used for analysis
tile_indices = np.array([[1, 1], [1, 23], [17, 1], [17, 23],
                         [7, 10], [7, 14], [11, 10], [11, 14]])

measured_wav_phase_path = r"data\experiments\2026-02-05_wavefront_measurement_test\run_2\phase_wavefront.npy"
measured_wav_amp_path   = r"data\experiments\2026-02-05_wavefront_measurement_test\run_2\amp_wavefront.npy"

measured_wav_phase = np.load(measured_wav_phase_path)
measured_wav_amp = np.load(measured_wav_amp_path)
incident_field = measured_wav_amp * np.exp(1j * measured_wav_phase)

# Create and run the app
app = SeidelMeasurementApp(exposure_time=exposure_time, 
                           exp_time_affine=exposure_time_affine,
                           tile_indices=tile_indices, 
                           first_last_tile_y=(first_tile_y, last_tile_y), 
                           coeff_range=coeff_range, 
                           coeff_step_size=coeff_step_size, 
                           field_inc=incident_field)
