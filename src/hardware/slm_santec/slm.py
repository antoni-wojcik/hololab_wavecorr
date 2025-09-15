# This script requires the Santec SLM _slm_win.py and the Santec SLM .dll files to be in the same directory as this script.

import ctypes
import numpy as np
from src.hardware.islm import ISLM  # Import the generic interface
import sys
if sys.platform.startswith("win"):
    from src.hardware.slm_santec import _slm_win as slmc  # Import the Santec SLM library
else:
    slmc = None
    print("Warning: Santec SLM not supported on non-Windows platforms.")
import time
import sys

# Constants
SLM_SHAPE = (1200, 1920)  # SLM shape in pixels
PIXEL_PITCH = 8 # Pixel pitch in micrometers
GRAYSCALE_RANGE = 1024 # Santec SLM supports 10-bit grayscale, so the range is 0-1023


# Define a class named 'SLM' (Spatial Light Modulator)
class SLMSantec(ISLM):
    def __init__(self, wavelength = 658, slm_number = 1, use_memory_mode = False, set_wavelength = True, timeout=10):
        '''
        This code interacts with a Spatial Light Modulator (SLM) to control its settings and display images on it.
        It first establishes a connection with the SLM, sets up its mode and wavelength (nm), and then displays images on it.
        The SLM is controlled using the Santec SLM library.
        '''
        # Define constants for different color flags
        self.FLAGS_COLOR_R    = 0x00000001 # Flag for Red color
        self.FLAGS_COLOR_G    = 0x00000002 # Flag for Green color
        self.FLAGS_COLOR_B    = 0x00000004 # Flag for Blue color
        self.FLAGS_COLOR_GRAY = 0x00000008 # Flag for grayscale
        self.FLAGS_RATE120    = 0x20000000 # Flag for 120Hz refresh rate

        # Define status codes
        self.SLM_OK = 0 # Status OK
        self.SLM_NG = 1 # Status Not Good
        self.SLM_BS = 2 # Status Bad State
        self.SLM_ER = 3 # Status Error

        # Set refresh rate
        self.Rate120 = 1 # Enable 120Hz refresh rate

        # Set maximum number of memory slots
        self.MAX_MEMORY = 128

        # Set display mode
        self.SLM_MODE_MEMORY = 0
        self.SLM_MODE_DVI = 1
        self.use_memory_mode = use_memory_mode

        # Set the wavelength and the SLM number
        self.wavelength = wavelength # The wavelength to be used by the SLM, in nanometers
        self.SLMNumber = slm_number # The number assigned to the SLM. Used if multiple SLMs are present.
        self.set_wavelength = set_wavelength
        self.timeout = timeout

        # Closing any previous instances of the SLM with that number:
        slmc.SLM_Ctrl_Close(self.SLMNumber)

    # Method to change the wavelength of the SLM
    def change_wavelength(self, wavelength, modulation_range=200, save=False):
        # Target modulation range = 2 * pi * moudlation_range / 200
        # Wavelength in nm
        print("Changing SLM calibration wavelength to", wavelength, "nm")
        slmc.SLM_Ctrl_WriteWL(self.SLMNumber, wavelength, modulation_range)

        if save:
            slmc.SLM_Ctrl_WriteAW(self.SLMNumber)

    # Method to connect the SLM
    def open(self, slm_display_number = 2):
        print('SLM connecting')
        self.width = ctypes.c_ushort(0)
        self.height = ctypes.c_ushort(0)
        self.DisplayName = ctypes.create_string_buffer(64)

        slm_found = False

        # Search LCOS-SLM
        for self.DisplayNumber in range(1, 8):
            ret = slmc.SLM_Disp_Info2(self.DisplayNumber, self.width, self.height, self.DisplayName)
            if (ret == self.SLM_OK):
                Names = self.DisplayName.value.decode('mbcs').split(',')
                
                if (Names[0] in 'LCOS-SLM' or self.DisplayNumber == slm_display_number):  # 'LCOS-SLM,SOC,8001,2018021001'
                    # Sometimes the name of the display may be wrong. 
                    # slm_display_number is used to override this by directly specifying which diplay is the SLM
                    self.width = self.width.value
                    self.height = self.height.value
                    self.shape = (self.height, self.width)
                    print(Names, self.width, self.height)
                    slm_found = True
                    break

        if (self.DisplayNumber >= 8 or not slm_found):
            print('No SLM')
            return

        # Set the rate of the SLM
        if (self.Rate120):
            self.Flags = self.FLAGS_RATE120
        else:
            self.Flags = 0

        # Open the USB connection to the SLM and initialise the SLM display
        slmc.SLM_Disp_Open(self.DisplayNumber)
        slmc.SLM_Ctrl_Open(self.SLMNumber)

        # Set the mode of the SLM (Memory or DVI)
        if self.use_memory_mode:
            slmc.SLM_Ctrl_WriteVI(self.SLMNumber, self.SLM_MODE_MEMORY)
        else:
            slmc.SLM_Ctrl_WriteVI(self.SLMNumber, self.SLM_MODE_DVI)

        # Change the wavelength of the SLM
        if self.set_wavelength:
            self.change_wavelength(self.wavelength)

        print('SLM connected')

    # Method to close the SLM
    def close(self):
        # Stop the SLM
        slmc.SLM_Ctrl_WriteDB(self.SLMNumber)
        # Close the USB connection to the SLM and close the SLM display
        slmc.SLM_Ctrl_Close(self.SLMNumber)
        slmc.SLM_Disp_Close(self.DisplayNumber)

        print('SLM closed')

    # Method to display an image on the SLM via DVI
    def display_dvi(self, values):
        '''
        Display the SLM values specified in the array. The array should have the dimensions of the SLM and uint16 or ushort data type.
        '''
        n_h, n_w = values.shape

        if n_w == self.width and n_h == self.height:
            values_flat = values.flatten().astype(np.ushort)
            # Unpack the list values_flat and pass its elements as individual arguments to the ctypes array constructor
            values_ctypes = (ctypes.c_ushort * len(values_flat))(*values_flat)
            # Display the hologram
            ret = slmc.SLM_Disp_Data(self.DisplayNumber, n_w, n_h, self.Flags, values_ctypes)
        else:
            print(f"Error: Wrong hologram dimensions: ({n_h}, {n_w})")
            pass

    # Method to display an image on the SLM via memory
    def display_memory(self, memory_offset):
        memory_number = 1 + memory_offset
        ret = slmc.SLM_Ctrl_WriteDS(self.SLMNumber, memory_number)
        if ret != self.SLM_OK:
            print(f"Error displaying memory slot {memory_number}")
            return
        
    # Combine both display functions into one based on self.use_memory_mode
    def display(self, values_or_memory_offset):
        """
        Display an image on the SLM. 
        If use_memory_mode is True, values_or_memory_offset is the memory slot number (0-127).
        If use_memory_mode is False, values_or_memory_offset is a 2D numpy array.
        Wait for the SLM to become ready before and after displaying for a maximum of timeout seconds.
        """
        self.wait_for_ready(timeout=self.timeout)
        if self.use_memory_mode:
            self.display_memory(values_or_memory_offset)
        else:
            self.display_dvi(values_or_memory_offset)

        self.wait_for_ready(timeout=self.timeout)

    def wait_for_ready(self, timeout=5):
        """
        Wait until the SLM is ready or timeout is reached.

        :param timeout: Maximum time to wait (in seconds)
        :return: True if SLM becomes ready, False if timeout occurs
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = slmc.SLM_Ctrl_ReadSU(self.SLMNumber)
            if status == self.SLM_OK:  # SLM is ready
                return True
            elif status == self.SLM_BS:  # SLM is busy
                time.sleep(0.01)  # Check every 10ms
            else:
                print(f"SLM error: status {status}")
                return False  # Exit early if error occurs

        print("WARNING: SLM did not become ready within the timeout period.")
        return True  # Timeout reached
    
    def load_memory(self, frames, memory_start=1):
        """
        Displays a sequence of 2D grayscale frames in memory mode.
        
        :param frames: List of 2D numpy arrays (grayscale images)
        :param memory_start: Memory number to start from (1-128)
        :param delay: Time delay between frames (in seconds)
        :param loop: Whether to loop indefinitely
        """
        
        if not frames:
            print("No frames to display")
            return

        self.num_frames_mem = len(frames)
        if self.num_frames_mem > self.MAX_MEMORY - memory_start + 1:
            print("Warning: Too many frames. Only first 128 slots will be used.")
        
        print(f"Loading {min(self.num_frames_mem, self.MAX_MEMORY)} frames into memory")

        # Load frames into memory
        for i, frame in enumerate(frames[:self.MAX_MEMORY]):
            print(f"Loading frame {i} into memory slot {memory_start + i}")
            
            memory_number = memory_start + i
            n_h, n_w = frame.shape

            if n_w != self.width or n_h != self.height:
                print(f"Skipping frame {i}: Incorrect dimensions {frame.shape}")
                continue

            frame_flat = frame.flatten().astype(np.ushort)
            frame_ctypes = (ctypes.c_ushort * len(frame_flat))(*frame_flat)

            ret = slmc.SLM_Ctrl_WriteMI(self.SLMNumber, memory_number, n_w, n_h, 0, frame_ctypes)
            if ret != self.SLM_OK:
                print(f"Error loading frame {i} into memory slot {memory_number}")
                return
            
        print(f"Loaded the frames into memory")

    @staticmethod
    def get_shape():
        return SLM_SHAPE
    
    @staticmethod
    def get_pixel_pitch():
        return PIXEL_PITCH
    
    @staticmethod
    def get_grayscale_range():
        """
        Returns the grayscale range for the SLM.
        """
        return GRAYSCALE_RANGE

    @staticmethod
    def phase_to_grayscale(phase):
        # Assume the phase is in the range [0, 2*pi]
        phase_wrapped = np.fmod(phase, 2 * np.pi)

        # Discretise the phase
        phase_discrete = np.round(phase_wrapped / (2 * np.pi) * GRAYSCALE_RANGE).astype(np.ushort) % GRAYSCALE_RANGE

        return phase_discrete  


# This condition checks if the script is being run directly or imported. If run directly, the code block under this condition will run.
if __name__ == '__main__':
    # Create an instance of the SLM class
    S = SLMSantec()
    S.open(2)
    S.close()

    print("It works!")
