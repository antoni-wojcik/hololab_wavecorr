# Script to measure the wavefront using the proposed method

from src.calibration.wavefront_meas import WavefrontMeasurer, MethodType
from src.hardware.slm_santec.slm import SLMSantec
from src.hardware.camera_thor.camera import CameraThorlabs
from src.io.experiment import ExpIO

from matplotlib import pyplot as plt

# Setup
wavelength = 633 # nm
grayscale_range = 1024
u0, v0 = 0.1, 0.1 # Location of the reference spot in the focal plane, range [-0.5, 0.5]
cam_exposure_time = 4e-4 # s
cam_gain = 0 # dB
slm_timeout = 5 # s
focal_length = 6 # cm
tile_size = 120

# Start logging the experiment
exp = ExpIO(name="wavefront_measurement_test")

# Open the SLM and camera and close them properly after the measurement or in case of an error
measurer = None
with SLMSantec(wavelength=wavelength, set_wavelength=False, timeout=slm_timeout) as slm, \
     CameraThorlabs(exposure_time=cam_exposure_time, gain=cam_gain) as camera:

    # Run the measurement
    measurer = WavefrontMeasurer(slm, camera, wavelength=wavelength, focal_length=focal_length, grayscale_range=grayscale_range, u0=u0, v0=v0, tile_size=tile_size)
    try:
        print("Running the wavefront measurement...")
        print("Press Ctrl+C in the terminal to stop the measurement.")
        measurer.run(extra_iterations = 5, method_type=MethodType.BROYSDEN, verbose=False, exp=exp)
    except KeyboardInterrupt:
        print("\nMeasurement stopped by the user.")
        exit()
    except Exception as e:
        raise e

print("Measurement finished successfully.")

# Save the calibration data
measurer.save(exp)

# Plot the amplitude at different segments
fig1 = measurer.plot_amp()
plt.show()

# Plot the fitted phase
fig2 = measurer.plot_phase()
plt.show()

# Plot the gradients and loss
fig3 = measurer.plot_gradients()
plt.show()

exp.save_figure(fig1, "amp_wav")
exp.save_figure(fig2, "phase_wav")
exp.save_figure(fig3, "segment_gradients")