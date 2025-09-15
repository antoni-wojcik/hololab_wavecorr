import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize
from src.hardware.islm import ISLM
from src.hardware.icamera import ICamera
from src.calibration.roi import ROISelector, ROI
from src.io.experiment import ExpIO
from tqdm import tqdm

ZERNIKE_NAMES = [
    "Piston",               # Z0
    "Tilt X",               # Z1
    "Tilt Y",               # Z2
    "Oblique Astigmatism",      # Z3
    "Defocus",              # Z4
    "Vertical Astigmatism",       # Z5
    "Vertical Trefoil",     # Z6
    "Vertical Coma",        # Z7
    "Horizontal Coma",      # Z8
    "Horizontal Trefoil",   # Z9
    "Oblique Quadrafoil",   # Z10
    "Oblique 2nd Astigmatism",  # Z11
    "Primary Spherical",    # Z12
    "Vertical 2nd Astigmatism",  # Z13
    "Vertical Quadrafoil",  # Z14
]

class Zernike:
    def __init__(self, slm_shape, num_zernikes = 15):
        self.shape = slm_shape
        self.num_zernikes = num_zernikes
        
        # Find the parameters used for calculation of the polynomials
        y, x = np.indices(slm_shape)
        half_h = slm_shape[0] // 2
        half_w = slm_shape[1] // 2
        y = y - half_h
        x = x - half_w
        self.r = np.sqrt(x**2 + y**2) / np.sqrt(half_w**2 + half_h**2)
        self.theta = np.arctan2(y, x)
        self.r[self.r > 1] = 0  # Mask out values outside the unit circle

        # Pre-calculate the polynomials
        self.poly_array = [None] * num_zernikes
        for k in range(num_zernikes):
            n, m = self.zernike_index(k)
            self.poly_array[k] = self.zernike_polynomial(n, m)

        # Store the names of the Zernike polynomials
        self.zernike_names = ZERNIKE_NAMES[:num_zernikes] if num_zernikes <= len(ZERNIKE_NAMES) else ZERNIKE_NAMES + [f"Z{k}" for k in range(len(ZERNIKE_NAMES), num_zernikes)]

    def zernike_index(self, k):
        """
        Map an index k to the Zernike polynomial indices (n, m).

        Parameters:
        - k (int): The index of the Zernike polynomial.

        Returns:
        - (n, m) (tuple): The radial and azimuthal order (n, m).
        """
        n = 0
        while k >= (n + 1):
            k -= (n + 1)
            n += 1
        m = -n + 2 * k
        return n, m

    def zernike_radial(self, n, m, r):
        """
        Calculate the Zernike radial polynomial R^m_n(r).
        
        Parameters:
        - n (int): Radial order.
        - m (int): Azimuthal order (|m| <= n, and (n - m) is even).
        - r (np.ndarray): Radial coordinates.

        Returns:
        - R_nm (np.ndarray): Radial polynomial evaluated at r.
        """
        R_nm = np.zeros_like(r)
        for k in range((n - m) // 2 + 1):
            R_nm += r**(n - 2*k) * (-1)**k * math.factorial(n - k) / \
                    (math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k))
        return R_nm

    def zernike_polynomial(self, n, m):
        """
        Generate the phase of the Zernike polynomial Z^m_n on a given shape.

        Parameters:
        - n (int): Radial order.
        - m (int): Azimuthal order (|m| <= n, and (n - m) is even).
        - shape (tuple): Shape of the output array (height, width).

        Returns:
        - Z_nm (np.ndarray): Zernike polynomial phase on the given shape.
        """

        if m >= 0:
            Z_nm = self.zernike_radial(n, m, self.r) * np.cos(m * self.theta)
        else:
            Z_nm = self.zernike_radial(n, -m, self.r) * np.sin(-m * self.theta)

        return Z_nm

    def get_phase(self, zernike_coeffs):
        """
        Generate the phase of Zernike polynomials from the coefficients.
        """
        # Generate the phase of zernike polynomials from the coefficients
        num_terms = len(zernike_coeffs) if self.num_zernikes > len(zernike_coeffs) else self.num_zernikes
        phase = np.sum([self.poly_array[k] * zernike_coeffs[k] for k in range(num_terms)], axis=0)

        return phase
    
    def fit_coefficients(self, target_phase):
        """
        Fit Zernike coefficients to a target phase.

        Parameters:
        - target_phase (np.ndarray): Target phase to fit.

        Returns:
        - coeffs (np.ndarray): Fitted Zernike coefficients.
        """
        def loss_function(coeffs):
            # Calculate the phase using the current coefficients
            calculated_phase = self.get_phase(coeffs)
            # Compute the mean squared error between the target and calculated phase
            return np.mean((target_phase - calculated_phase) ** 2)

        # Initial guess for the coefficients (zeros)
        initial_guess = np.zeros(self.num_zernikes)

        # Perform the minimization
        result = minimize(loss_function, initial_guess, method='L-BFGS-B')

        # Return the optimized coefficients
        return result.x
    
    def fit_coefficients_sequential(self, target_phase):
        """
        Fit Zernike coefficients to a target phase sequentially.

        Parameters:
        - target_phase (np.ndarray): Target phase to fit.

        Returns:
        - coeffs (np.ndarray): Fitted Zernike coefficients.
        """
        coeffs = np.zeros(self.num_zernikes)
        for k in range(self.num_zernikes):
            def loss_function(c):
                coeffs[k] = c
                calculated_phase = self.get_phase(coeffs)
                return np.mean((target_phase - calculated_phase) ** 2)

            # Initial guess for the coefficient (zero)
            initial_guess = 0.0

            # Perform the minimization
            result = minimize(loss_function, initial_guess, method='L-BFGS-B')

            # Store the optimized coefficient
            coeffs[k] = result.x

        return coeffs
    
class ZernikeMeasurer:
    def __init__(self, slm: ISLM, camera: ICamera, u0, v0, num_zernikes=15, grayscale_range=1024):
        self.slm_shape = slm.get_shape()
        self.M, self.N = self.slm_shape
        self.u0 = u0
        self.v0 = v0
        self.grayscale_range = grayscale_range
        self.deflector_phase = self._get_deflector_phase(u0, v0)
        self.zernike = Zernike(self.slm_shape, num_zernikes)

        self.slm = slm
        self.camera = camera
        self.roi: ROI = None

    def _psf_loss(self, psf_capture, mask_radius=8, verbose=False):
        res = psf_capture.shape
        mask_center = [self.com[1], self.com[0]]  # [x, y] -> [col, row]
        X, Y = np.meshgrid(np.arange(res[0]), np.arange(res[1]), indexing='ij')
        # We generate a mask representing the disk we want to intensity to be concentrated in
        mask = (X - mask_center[0]) ** 2 + (Y - mask_center[1]) ** 2 < mask_radius ** 2
        signal = np.sum(psf_capture * mask) / np.sum(mask)
        noise = np.sum(psf_capture * (1 - mask)) / np.sum(1 - mask)
        cost = signal / noise

        if verbose:
            plt.imshow(psf_capture, cmap='gray')
            plt.imshow(mask, cmap='Reds', alpha=0.5)
            plt.title("PSF Capture with Mask")
            plt.show()

        return cost

    def run(self, coeff_range, num_steps=20, mask_radius=8, sweep=True):
        self.zernike_coeffs = np.zeros(self.zernike.num_zernikes)

        # Display the patterns
        pattern = self._quantise_phase(self.deflector_phase)
        self.slm.display(pattern)
        capture = self.camera.capture()
        self._select_roi(capture)

        # Set the camera ROI
        # THIS PRODUCES ERRORS FOR SOME REASON - THE WIDTH AND HEIGHT ARE NOT SET PROPERLY
        self.roi.set_camera_roi(self.camera)

        capture = self.camera.capture()
        self.com = self.roi.get_com(capture)

        initial_loss = self._psf_loss(capture, mask_radius, verbose=True)
        print(f"Initial loss: {initial_loss}")
        
        # Iterate over the Zernike coefficients (excluding piston, tip, and tilt)
        for i in tqdm(range(3, self.zernike.num_zernikes), desc="Zernike Analysis", unit="zernike"):
            
            if sweep:
                min_loss = float('inf')
                best_coeff = 0

                bar_coeff = tqdm(np.linspace(-coeff_range, coeff_range, num_steps), desc=f"Zernike Coeff {i}", unit="coeff")

                # Iterate over the coefficient range
                for i_coeff in bar_coeff:
                    self.zernike_coeffs[i] = i_coeff

                    # Get the phase
                    phase_zernike = self.zernike.get_phase(self.zernike_coeffs)
                    phase = phase_zernike + self.deflector_phase
                    pattern = self._quantise_phase(phase)
                    
                    # Display the phase on the SLM
                    self.slm.display(pattern)

                    # Capture the image
                    capture = self.camera.capture()

                    # Evaluate the PSF
                    loss = self._psf_loss(capture, mask_radius)

                    # Check if the loss is lower than the minimum loss
                    if loss < min_loss:
                        min_loss = loss
                        best_coeff = i_coeff

                    # Update the progress bar with the current loss
                    bar_coeff.set_postfix_str(f"Loss: {loss:.2f}")
            else:
                # Use a minimization algorithm to find the best coefficient
                def loss_function(c):
                    self.zernike_coeffs[i] = c
                    phase_zernike = self.zernike.get_phase(self.zernike_coeffs)
                    phase = phase_zernike + self.deflector_phase
                    pattern = self._quantise_phase(phase)

                    # Display the phase on the SLM
                    self.slm.display(pattern)

                    # Capture the image
                    capture = self.camera.capture()

                    # Evaluate the PSF
                    loss = self._psf_loss(capture, mask_radius)
                    print(f"Loss: {loss:.2f} for coeff: {c} (Zernike {i})")

                    return loss
                
                # Initial guess for the coefficient
                initial_guess = 10.0

                # Perform the minimization
                result = minimize(loss_function, initial_guess, method='Nelder-Mead')
                best_coeff = result.x[0]
                min_loss = result.fun

            # Store the best coefficient
            self.zernike_coeffs[i] = best_coeff

            phase_zernike = self.zernike.get_phase(self.zernike_coeffs)
            phase = phase_zernike + self.deflector_phase
            pattern = self._quantise_phase(phase)

            # Display the phase on the SLM
            self.slm.display(pattern)

            # Capture the image
            capture = self.camera.capture()

            plt.imshow(capture, cmap='gray')
            plt.title(f"Zernike {i} - Best Coefficient: {best_coeff:.2f}, Loss: {min_loss:.2f}")
            plt.colorbar()
            plt.show()

            print(f"Best coefficient for Zernike {i}: {best_coeff} with loss: {min_loss}")

        print("Fitting complete.")

        return self.zernike_coeffs
    
    def save(self, exp: ExpIO):
        """
        Save the calibration data to the experiment object.
        """
        # Save the Zernike coefficients
        exp.save_npy(self.zernike_coeffs, "zernike_coeffs")

        # Save the phase
        phase_zernike = self.zernike.get_phase(self.zernike_coeffs)
        exp.save_npy(phase_zernike, "phase_zernike")

        # Save the deflector hologram
        phase = phase_zernike + self.deflector_phase
        pattern = self._quantise_phase(phase)
        exp.save_hologram(pattern, "hologram_zernike")

    def _select_roi(self, camera_image):
        camera_selector = ROISelector(camera_image, window_name="Camera Image")
 
        ROISelector.print_instructions()
 
        while True:
            camera_selector.update()
 
            if camera_selector.check_esc_key(): # Press 'Esc' to exit selection
                camera_selector.close()
                break
 
        if len(camera_selector.rois) < 1:
            print("No point selected. Please restart and select at least 1.")
            exit()
 
        # Get only the first ROI
        self.roi = camera_selector.rois[0]
            
    def _quantise_phase(self, phase):
        phase_wrapped = np.mod(phase, 2 * np.pi)
        grayscale_values = np.round(phase_wrapped / (2 * np.pi) * self.grayscale_range).astype(np.ushort) % self.grayscale_range
        return grayscale_values
            
    def _get_deflector_phase(self, u0, v0):
        grad_x = 2 * np.pi * u0 * 0.5
        grad_y = 2 * np.pi * v0 * 0.5
        # Get the zernike tilt phase
        x, y = np.linspace(0, self.N, self.N, endpoint=False), np.linspace(0, self.M, self.M, endpoint=False)
        m, n = np.meshgrid(y, x, indexing='ij')
        phase = grad_y * m + grad_x * n # m is the y-axis, n is the x-axis

        return phase
    
    def plot_coeffs(self):
        """
        Plot the Zernike coefficients.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(self.zernike.num_zernikes), self.zernike_coeffs)
        ax.set_xlabel('Zernike Coefficient Index')
        ax.set_ylabel('Coefficient Value')
        ax.set_title('Zernike Coefficients')
        
        return fig
    
    def plot_phase(self):
        """
        Plot the Zernike phase.
        """
        phase = self.zernike.get_phase(self.zernike_coeffs)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(phase, cmap='gray')
        plt.title('Zernike Phase')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        return fig
                
    
if __name__ == "__main__":
    # Example usage
    slm_shape = (1200, 1920)
    zernike = Zernike(slm_shape, num_zernikes=15)
    
    # Generate Zernike coefficients
    zernike_coeffs = np.random.rand(zernike.num_zernikes) * 10
    
    # Get the phase of the Zernike polynomials
    phase = zernike.get_phase(zernike_coeffs)

    # Visualize the results
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    im = ax.imshow(phase, cmap='gray')
    plt.title('Zernike Phase')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()

    # phase = np.load(r"data/experiments/2025-04-10_wavefront_measurement_test/run_0/phase_wavefront.npy")

    # # Fit Zernike coefficients to a target phase
    # target_phase = np.copy(phase)  # Example target phase
    # fitted_coeffs = zernike.fit_coefficients(target_phase)
    
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].imshow(target_phase, cmap='gray')
    # axs[0].set_title('Target Phase')
    # axs[1].imshow(zernike.get_phase(fitted_coeffs), cmap='gray')
    # axs[1].set_title('Fitted Phase')
    # plt.show()

    # print("Fitted Coefficients:", fitted_coeffs)