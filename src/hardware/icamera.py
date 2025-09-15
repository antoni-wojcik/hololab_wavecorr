from abc import ABC, abstractmethod

class ICamera(ABC):
    """ Abstract Base Class for Scientific Cameras. """

    @abstractmethod
    def open(self):
        """ Establish connection to the camera. """
        pass

    @abstractmethod
    def close(self):
        """ Close the connection to the camera. """
        pass

    @abstractmethod
    def capture(self):
        """ Capture a picture with the camera. """
        pass

    @abstractmethod
    def capture_hdr(self, num_frames):
        """ Capture a high dynamic range picture with the camera. """
        pass

    @staticmethod
    @abstractmethod
    def get_range():
        """ Return the range of the camera in pixels. """
        pass

    @staticmethod
    @abstractmethod
    def get_pixel_pitch():
        """ Return the pixel pitch of the camera in microns. """
        pass
    
    @staticmethod
    @abstractmethod
    def get_shape():
        """ Return the shape (resolution) of the camera. """
        pass

    @abstractmethod
    def set_roi(self, x, y, width, height):
        """ Set the region of interest for the camera. """
        pass

    def __enter__(self):
        """ Ensure the device is opened when entering a with-block. """
        self.open()
        return self  # Allows the object to be used inside the 'with' block

    def __exit__(self, exc_type, exc_value, traceback):
        """ Ensure the device is closed when exiting a with-block. """
        self.close()