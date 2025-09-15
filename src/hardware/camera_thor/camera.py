# The original code for the Thorlabs cameras breaks when used with monochrome cameras. The script TLCamera.py needs to be updated with the code:
# https://github.com/AlexShkarin/pyLabLib/issues/65
from src.hardware.icamera import ICamera
import pylablib.devices.Thorlabs as tl
import numpy as np

# Constants
CAMERA_SHAPE = (1080, 1440)  # Camera shape in pixels
PIXEL_PITCH = 3.45 # Pixel pitch in micrometers
GRAYSCALE_RANGE = 1024 # 10 bit grayscale range (0-1023)


class CameraThorlabs(ICamera):
    def __init__(self, camera_num = 0, exposure_time = 0.01, gain = 0, flip_ud = True, flip_lr = False):
        self.camera_num = camera_num
        self.exposure_time = exposure_time # Exposure time in seconds
        self.gain_db = gain # Gain in dB

        self.width, self.height = 0, 0

        # Constants
        self.CAPTURE_TIMEOUT = 15 # Set the maximum timeout on a camera capture to 15 s

        # Flip the image
        self.flip_ud = flip_ud
        self.flip_lr = flip_lr

        # Initialize camera
        self.camera = None

    # Initialize the camera
    def open(self):
        detected_camera = tl.list_cameras_tlcam()[self.camera_num]
        self.camera = tl.ThorlabsTLCamera(serial=detected_camera)
        self.camera.open()

        # Verify if the camera opened successfully and print the status
        print('Camera opened:', self.camera.is_opened())

        # Switch the LED indicator off and reset the settings of the camera
        self.camera.enable_led(False)

        # Set the camera settings
        self.set_gain(self.gain_db)
        self.set_exposure_time(self.exposure_time)

        # Get the detector size
        self.width, self.height = self.camera.get_detector_size()

        # Reset the camera settings
        self.set_roi(0, 0, width=self.width, height=self.height)

    # Close the camera
    def close(self):
        # Reset the camera settings
        self.set_roi(0, 0, width=self.width, height=self.height)

        self.camera.close()

        # Verify if the camera closed successfully and print the status
        print('Camera closed:', not self.camera.is_opened())
    
    # Take a snapshot
    def capture(self):
        frame = self.camera.snap(timeout=self.CAPTURE_TIMEOUT).astype(np.double)

        if self.flip_ud:
            frame = np.flipud(frame)
        if self.flip_lr:
            frame = np.fliplr(frame)
        
        return frame
    
    # Take a HDR snapshot
    def capture_hdr(self, num_frames):
        images = np.zeros((num_frames, self.height, self.width), dtype=np.float32)
        exposure_times = np.zeros(num_frames, dtype=np.float32)

        exposure_time_target = self.exposure_time

        # Take a series of images with different exposure times in powers of 10
        for i in range(num_frames):
            exposure_times[i] = self.camera.get_exposure()
            images[i, :, :] = self.capture().astype(np.float32)
            exposure_time_target *= 10
            self.camera.set_exposure(exposure_time_target)

        self.camera.set_exposure(self.exposure_time)  # Reset the exposure time

        # Compute weight map: Higher exposure images contribute more in dark areas
        weights = np.clip(images, 1, GRAYSCALE_RANGE - 1)  # Avoid division by zero
        weights = 1.0 / weights  # Lower values get higher weights

        # Compute HDR image
        numerator = np.sum(weights * images / exposure_times[:, None, None], axis=0)
        denominator = np.sum(weights / exposure_times[:, None, None], axis=0)
        
        hdr = numerator / denominator  # Final HDR image

        return hdr / np.max(hdr)  # Normalize to [0, 1]
        
    # Set the exposure time
    def set_exposure_time(self, exposure_time):
        # Set the exposure time in seconds (Zelux CS165MU1/M)
        # Might be ms or us for other camera models
        self.camera.set_exposure(exposure_time)

    # Set the gain
    def set_gain(self, gain_db):
        self.camera.set_gain(gain_db)

    # Other methods specific to the Thorlabs camera
    def start_acquisition(self):
        self.camera.start_acquisition()
    
    def stop_acquisition(self):
        self.camera.stop_acquisition()
    
    def get_frame(self):
        self.camera.wait_for_frame()  # wait for the next available frame
        frame = self.camera.read_oldest_image(timeout=self.CAPTURE_TIMEOUT).astype(np.double)
        return frame

    @staticmethod
    def get_range():
        return GRAYSCALE_RANGE - 1

    @staticmethod
    def get_pixel_pitch():
        return PIXEL_PITCH
    
    @staticmethod
    def get_shape():
        return CAMERA_SHAPE
    
    def set_roi(self, x, y, width, height, verbose=False):
        if self.flip_lr:
            x = self.width - x - width
        if self.flip_ud:
            y = self.height - y - height

        if verbose:
            print("Setting Camera ROI to x={}, y={}, width={}, height={}".format(x, y, width, height))

        self.camera.set_roi(hstart=x, hend=x + width, vstart=y, vend=y + height)
    
    # def set_roi(self, x, y, width, height, verbose=False):
    #     """
    #     Sets ROI; respects the increment constraints.
    #     """
    #     if self.camera is None or not self.camera.is_opened():
    #         return

    #     hlim, vlim = self.camera.get_roi_limits()
    #     hmin, hmax, hpstep, hsstep, hmaxbin = hlim
    #     vmin, vmax, vpstep, vsstep, vmaxbin = vlim

    #     # Adjust x and y based on flip settings
    #     if self.flip_lr:
    #         x = self.width - x - width
    #     if self.flip_ud:
    #         y = self.height - y - height

    #     # Ensure x and y respect the position increments
    #     x = (x // hpstep) * hpstep
    #     y = (y // vpstep) * vpstep

    #     # Ensure width and height respect the size increments
    #     width = (width // hsstep) * hsstep
    #     height = (height // vsstep) * vsstep

    #     # Ensure x, y, width, and height are within valid limits
    #     x = max(hmin, min(hmax - width, x))
    #     y = max(vmin, min(vmax - height, y))
    #     width = max(hsstep, min(hmax - x, width))
    #     height = max(vsstep, min(vmax - y, height))

    #     if verbose:
    #         print("Setting Camera ROI to x={}, y={}, width={}, height={}".format(x, y, width, height))

    #     self.camera.set_roi(hstart=x, hend=x + width, vstart=y, vend=y + height)
