from abc import ABC, abstractmethod

class ISLM(ABC):
    """ Abstract Base Class for Spatial Light Modulators """

    @abstractmethod
    def open(self):
        """ Establish connection to the SLM device. """
        pass

    @abstractmethod
    def close(self):
        """ Close the connection to the SLM. """
        pass

    @abstractmethod
    def display(self, values):
        """ Display an image on the SLM. """
        pass

    @staticmethod
    @abstractmethod
    def get_shape(self):
        """ Return the shape (resolution) of the SLM. """
        pass

    @staticmethod
    @abstractmethod
    def get_pixel_pitch(self):
        """ Return the pixel pitch of the SLM in microns. """
        pass

    def __enter__(self):
        """ Ensure the device is opened when entering a with-block. """
        self.open()
        return self  # Allows the object to be used inside the 'with' block

    def __exit__(self, exc_type, exc_value, traceback):
        """ Ensure the device is closed when exiting a with-block. """
        self.close()