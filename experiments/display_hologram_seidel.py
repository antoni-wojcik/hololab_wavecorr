from src.cgh.propagator import PropagatorTorch, AberrCoefficients, SeidelPropagatorSeries, SeidelPropagatorPatches
from src.cgh.optimisers import PhaseOptimBGD
from src.cgh.patterns import make_cross_square_tiled
from src.io.img_handler import load_svg
from src.cgh.utils import squish_image
from src.hardware.slm_santec.slm import SLMSantec
from src.hardware.camera_thor.camera import CameraThorlabs
from src.io.experiment import ExpIO
import matplotlib.pyplot as plt
import torch
import numpy as np
from time import time

# This script tests the display of holograms optimized with wavefront curvature compensation using the Seidel propagators. The incident wavefront needs to be measured beforehand and the aberration coefficients for the Seidel correction need to be obtained from a previous experiment. The script can load a target intensity pattern from an SVG file or generate a tiled cross marker pattern as the target. The optimization is performed using Time-Multiplexed Batched Gradient Descent (TMBGD) for multiple batches of holograms, which are then displayed on the SLM and captured with a camera. The experiment is repeated for different types of Seidel propagators (tiled, series expansion, and standard) for comparison. The results, including the captured images and convergence history, are saved for analysis.

wavelength = 633 # nm
cam_gain = 0 # camera gain in dB
scale = 2  # Scale factor for the propagator
batch_size = 4 # Number of holograms to calculate in parallel
num_iterations = 60 # Number of iterations for the optimization algorithm
sub_frames = 4  # Number of sub-frames for the SLM. If > 1, the SLM will display multiple sub-frames for each hologram obtained by offsetting the grayscale values. This improves the time averaged results.
learning_rate = 0.2  # Learning rate for the optimizer
exposure_time_full = 0.25e-1 # s

# Load the aberration coefficients obtained from a previous experiment. The coefficients should be in the order: (coma, astigmatism, field curvature, distortion, tip, tilt). 
seidel_coeff_path = r"data/experiments/2026-02-09_wavefront_manual_obs/run_sweep_second_dist/optimal_seidel_coeffs.npy"
aberr = AberrCoefficients(*np.load(seidel_coeff_path))

# Possible propagator types: "tiled" for the tiled Seidel propagator, "series" for the series expansion Seidel propagator, "standard" for the standard propagator without Seidel aberration correction. A list can be provided to run the experiment with multiple propagator types for comparison. 
prop_types = ["tiled", "series", "standard"] # Run the experiment with all the propagator types for comparison.

# If tiling (patch) method used:
num_tiles = 20 # Number of tiles in the Seidel propagator
boundary = 30  # Boundary for the Seidel propagator
blending = False # Whether to use blending between neighboring tiles
vectorise_prop = False # Whether to vectorise the propagator calculations for performance when using the tiled seidel propagator (slower on a GPU that I tested, but faster on a CPU).

# If series expansion method used, calculate the number of terms automatically
num_terms = 30
use_chebyshev_approx = True # Whether to apply Chebyshev approximation to the series expansion. Otherise the Taylor series expansion is used. 

# Define the target path or generate a target pattern
load_target = True
target_path = r"data\targets\tmbgd_target_square.svg"  # Path to the target SVG file
marker_size = 30  # Size of the markers in the target pattern

# If loading precomputed data, set the paths for the wavefront phase and amplitude data
load_wav_data = True
measured_wav_phase_path = r"data\experiments\2026-02-05_wavefront_measurement_test\run_2\phase_wavefront.npy"
measured_wav_amp_path   = r"data\experiments\2026-02-05_wavefront_measurement_test\run_2\amp_wavefront.npy"


# -------------------------------------------------------------------
# Run the experiment with the provided parameters
# -------------------------------------------------------------------
NUM_TESTS = 3 # Number of different holograms to test. Each hologram will have a different portion of the target intensity pattern visible to decouple the effects of crosstalk and aberration correction.

if load_wav_data:
    measured_wav_phase = np.load(measured_wav_phase_path)
    measured_wav_amp = np.load(measured_wav_amp_path)
    incident_field = measured_wav_amp * np.exp(1j * measured_wav_phase)
else:
    # Use a uniform incident field
    incident_field = None

# If using a GPU, clear the cache to free up memory
if torch.cuda.is_available():
    torch.cuda.empty_cache() 

slm_shape = SLMSantec.get_shape()  # Get the SLM shape from the SLMLuna class

for i, prop_type in enumerate(prop_types):
    # Start logging the experiment
    exp = ExpIO(name=f"seidel_display", 
                suffix=prop_type,
                description=f"""Testing Time-Multipliexed Batched Gradient Descent (TMBGD) with wavefront curvature compensation.
                This experiment displays holograms on the SLM and captures the results with a camera.
                The holograms are generated using a curvature wavefront propagator and optimized using TMBGD.
                The target intensity is a tiled cross marker pattern.

                This method used the Seidel correction of type: "{prop_type}".
                """)
    
    print("Running experiment with propagator type:", prop_type)

    cam_exposure_times = [0] * NUM_TESTS # Calculate exposure times based on the total power of the target intensity pattern for each test to ensure equal saturation levels in the captured images.

    start_time = time()
    if prop_type == "tiled":
        prop = SeidelPropagatorPatches(slm_shape, aberr=aberr, field_inc=incident_field, scale=scale, num_tiles=num_tiles, bound=boundary, use_blending=blending, vectorise=vectorise_prop)
    elif prop_type == "series":
        prop = SeidelPropagatorSeries(slm_shape, aberr=aberr, field_inc=incident_field, scale=scale, high_precision=not use_chebyshev_approx, power_threshold=power_threshold, num_terms=num_terms, chebyshev_approx=use_chebyshev_approx)
    else:
        prop = PropagatorTorch(slm_shape, field_inc=incident_field, scale=scale)
    end_time = time()

    print(f"Initialized the propagator {prop_type} in {end_time - start_time:.2f} seconds.")

    if load_target:
        # Load the target intensity pattern from the SVG file
        max_dim = max(*prop.get_far_shape())
        int_target = load_svg(target_path, size=(max_dim, max_dim))
        int_target = squish_image(int_target, target_shape=prop.get_far_shape())
    else:
        int_target = make_cross_square_tiled(target_shape=prop.get_far_shape(), num_markers=np.gcd(*prop.get_slm_shape()) // 10, marker_size=marker_size)
        # int_target = np.ones(prop.get_far_shape())

    # Generate 3 batches of holograms: 
    # - the full target,
    # - just left side of the target, 
    # - just right side of the target, 
    # The first two decouple the effects of crosstalk from aberration correction which makes the analysis clearer. 
    # The third one shows the full hologram with all the defects present.
    grayscale_value_batches = [None] * NUM_TESTS

    for i in range(NUM_TESTS):
        # Prepare the target intensity for this test.
        if i == 0:
            int_target_test = int_target # Full target
        elif i == 1:
            int_target_test = np.copy(int_target)
            int_target_test[:, int_target.shape[1]//2:] = 0 # Just left side of the target
        else:
            int_target_test = np.copy(int_target)
            int_target_test[:, :int_target.shape[1]//2] = 0 # Just right side of the target

        # Calculate exposure time for the target
        cam_exposure_times[i] = np.sum(int_target_test) * exposure_time_full / np.sum(int_target) # Scale the exposure time based on the total power of the target to keep the captured images well-exposed.

        start_time = time()
        # Prepare the optimiser
        optim = PhaseOptimBGD(prop, target_int=int_target_test, num_iter=num_iterations, batch_size=batch_size, precalc_near_field=True)

        phase_init = np.random.rand(batch_size, *slm_shape) * 2 * np.pi  # Random initial phase
        phase_final = optim.optim(phase_init, learning_rate=learning_rate)
        end_time = time()

        print(f"Optimized the phase patterns propagator {prop_type} for test {i} in {end_time - start_time:.2f} seconds.")

        # Get the final simulated far-field
        int_far = prop.get_intensity(optim.field_far).cpu().numpy()

        # Separate the grayscale values for each hologram
        grayscale_values = [None] * batch_size
        for j in range(batch_size):
            # Convert the phase to grayscale values
            grayscale_values[j] = SLMSantec.phase_to_grayscale(phase_final[j])

        # Get the convergence history and save it
        fig = plt.figure(figsize=(10, 5), dpi=200)
        plt.plot(np.arange(num_iterations), optim.loss_history, label='Loss')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        exp.save_figure(fig, f"loss_history_{i}")

        # Save the grayscale values as holograms
        grayscale_range = SLMSantec.get_grayscale_range() # Grayscale range for the SLM
        for j in range(batch_size):
            exp.save_npy(grayscale_values[j], f"holo_{j}_target_{i}")
            exp.save_hologram(grayscale_values[j], f"holo_{j}_target_{i}")

        # Save the target intensity
        exp.save_image(int_target_test, f"int_target_{i}")

        # Save the predicted intensity
        exp.save_image(int_far, f"int_far_predicted_{i}")
        exp.save_npy(int_far, f"int_far_predicted_{i}")

        # Store the final grayscale values for this test
        grayscale_value_batches[i] = grayscale_values


    # Open the SLM and camera using context managers
    with SLMSantec(wavelength=wavelength, use_memory_mode=False, set_wavelength=False) as slm, \
        CameraThorlabs(exposure_time=exposure_time_full, gain=cam_gain) as camera:

        for k in range(NUM_TESTS):

            start_time = time()

            # Set the camera exposure time for this test
            camera.set_exposure_time(cam_exposure_times[k])

            # Display the final phase pattern for each test as a hologram on the SLM and capture the results with the camera. If sub-frames are used, display multiple sub-frames for each hologram by offsetting the grayscale values and capture each one.

            if sub_frames < 2:
                # Display the holograms and capture the results
                captures = [None] * batch_size
                for i in range(batch_size):
                    grayscale_value = grayscale_value_batches[k][i]
                    print(f"Displaying hologram {i} for target {k}")

                    slm.display(grayscale_value)
                    captures[i] = camera.capture()

                    # Save the results
                    exp.save_npy(captures[i], f"capture_{i}_target_{k}")
                    exp.save_image(captures[i], f"capture_{i}_target_{k}")

                    print(f"Saved results for hologram {i} for target {k}")
            else:
                grayscale_range = SLMSantec.get_grayscale_range() # Grayscale range for the SLM

                # Display the holograms with sub-frames and capture the results
                captures = [None] * batch_size * sub_frames
                for i in range(batch_size):
                    grayscale_value = grayscale_value_batches[k][i]
                    for j in range(sub_frames):
                        # Add a wrapped sub-frame offset to the grayscale value
                        grayscale_value_sub = np.ushort(np.float32(grayscale_value) + np.float32(grayscale_range) * j / sub_frames) % grayscale_range
                        print(f"Displaying hologram {i}, sub-frame {j} for target {k}")

                        slm.display(grayscale_value_sub)
                        captures[i * sub_frames + j] = camera.capture()

                        print(f"Saved results for hologram {i}, sub-frame {j} for target {k}")

                # Average the captures
                avg_capture = np.mean(np.array(captures), axis=0)

                # Save the average capture
                exp.save_npy(avg_capture, f"avg_capture_target_{k}")
                exp.save_image(avg_capture, f"avg_capture_target_{k}")

                print(f"Saved the average capture for target {k}")

            end_time = time()
            print(f"Displayed the holograms and captured the results for propagator {prop_type}, target {k} in {end_time - start_time:.2f} seconds.")

    print(f"Experiment completed successfully for propagator {prop_type}. All results saved.")