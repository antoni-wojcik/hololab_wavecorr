# Wavefront measurement using a tiled grating approach
# The script also includes classes to fit the amplitude and phase of the measured wavefront
# The GLIP (Gradient Linear Interpolation and Poisson reconstruction) method is implemented for phase fitting, and is the method used in the paper

from src.hardware.islm import ISLM
from src.hardware.icamera import ICamera
 
from src.calibration.roi import ROISelector, ROI
from src.io.experiment import ExpIO
 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from scipy.optimize import minimize
from scipy.fftpack import dct, idct
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from enum import Enum

class MethodType(Enum):
    BROYSDEN = 1
    SECANT = 2
    TRIVIAL = 3

class PhaseApproxMethod(Enum):
    GLIP = 1 # Bilinear interpolation of the gradient values, followed by Poisson reconstruction. This method is used in the paper
    MESH = 2
    POISSON = 3
 
class WavefrontMeasurer:
    def __init__(self, slm: ISLM, camera: ICamera, wavelength = 658, focal_length = 15, grayscale_range = 1024, u0=0.1, v0=0.1, tile_size = -1):
        self.slm = slm
        self.camera = camera
        self.slm_shape = slm.get_shape()
        self.wavelength = wavelength # nm
        self.focal_length = focal_length # cm
        self.grayscale_range = grayscale_range
 
        if tile_size <= 0:
            tile_size = np.gcd(*self.slm_shape)
            self.tile_positions = self.slm_shape // tile_size
        else:
            self.tile_positions = np.array([self.slm_shape[0] // tile_size, self.slm_shape[1] // tile_size])

        self.tile_size = tile_size
 
        # Create the grating tile to deflect at the angle like a grating of period = period. u_0 is between [-0.5, 0.5] which describes the first order diffraction
        self.grad_ref_x = 2 * np.pi * u0
        self.grad_ref_y = 2 * np.pi * v0
        self.gr_tile = self._get_tilted_tile(grad_x=self.grad_ref_x, grad_y=self.grad_ref_y)
        # Scaling constant to convert the gradients to radians per SLM pixel
        self.scaling_const = self.camera.get_pixel_pitch() * self.slm.get_pixel_pitch() * 2 * np.pi / (self.focal_length * self.wavelength) * 0.1
        print("Scaling constant: ", self.scaling_const)
 
        # Initial values
        self.roi: ROI = None
        self.amp_fit = None
        self.phase_fit = None
        self.tile_ints = None
        self.amp_wav = None
        self.phase_wav = None
        self.gradients = None # In radians per SLM pixel
 
    def run(self, extra_iterations = 1, method_type:MethodType = MethodType.BROYSDEN, phase_approx_method:PhaseApproxMethod = PhaseApproxMethod.GLIP, damping = 0.8, verbose = False, exp:ExpIO = None):
        # Create the grating pattern using the tilted tile
        pattern = self._place_tile(self.gr_tile)
 
        # Display the patterns
        self.slm.display(pattern)
 
        # Capture the image
        capture = self.camera.capture()
 
        # Select the ROI
        self._select_roi(capture)

        # Analyse the ROI
        if not verbose:
            self.roi.set_camera_roi(self.camera)
        capture = self.camera.capture()
        self.roi.analyse(capture, detect_blobs=True)
        central_roi_com = self.roi.com

        print("ROI size: ", self.roi.roi.shape)
        print("Refrence centre of mass: ", central_roi_com)

        if exp is not None:
            self.save_rois = False
            capture_roi = np.copy(self.roi.roi)
            capture_thres = np.minimum(capture_roi, 1024 * 0.5)
            exp.save_image(capture_thres, f"roi_reference")
            exp.save_npy(capture_roi, f"roi_reference")
        
        def _get_shift(grad, tile_pos):
            # Get the tilted tile and place it on the SLM, then display the pattern and capture the image
            tile = self._get_tilted_tile(grad[0] + self.grad_ref_x, grad[1] + self.grad_ref_x)
            # tile = self.gr_tile + tilt
            pattern = self._place_tile(tile=tile, tile_pos=tile_pos)
            self.slm.display(pattern)
            capture = self.camera.capture()

            # Analyse the ROI
            self.roi.analyse(capture, detect_blobs=True)
            current_com = self.roi.com

            # Calculate the shift of the centre of mass
            shift = current_com - central_roi_com

            if self.save_rois:
                capture_roi = np.copy(self.roi.roi)
                capture_thres = np.minimum(capture_roi, 1024 * 3e-2)
                exp.save_image(capture_thres, f"roi{tile_pos[0]}{tile_pos[1]}", separate_dir=True)
                exp.save_npy(capture_roi, f"roi{tile_pos[0]}{tile_pos[1]}", separate_dir=True)
                text = f"""Tile {tile_pos}:
Gradient: {grad / (2 * np.pi)}
Central COM: {central_roi_com}
Current COM: {current_com}
Current local COM: {self.roi.com_local}
Shift: {shift}
ROI size: {self.roi.roi.shape}
                """
                exp.save_text(text, f"roi{tile_pos[0]}{tile_pos[1]}", separate_dir=True)

            # Display image seen by the camera
            # if verbose:
            #     plt.imshow(capture_thres, cmap='gray')
            #     plt.colorbar()
            #     plt.show()

            return shift
 
        # Initialize tqdm progress bar
        total_iterations = self.tile_positions[0] * self.tile_positions[1]
        tqdm_bar = tqdm(
            total=total_iterations,  # Total iterations
            desc="Wavefront Measurement",
            unit="iter",
            leave=True,
            bar_format="{l_bar}{bar} [iter: {n_fmt}/{total_fmt}, time left: {remaining}, time spent: {elapsed}{postfix}]",
            position=0
        )

        # Go over all the tiles to measure the amplitude
        self.tile_ints = np.zeros(self.tile_positions)
        self.gradients = np.zeros((self.tile_positions[0], self.tile_positions[1], 2))
        num_steps_final = np.zeros(self.tile_positions, dtype=int)

        for i in range(self.tile_positions[0]):
            for j in range(self.tile_positions[1]):
                tile_pos = (i, j)

                if verbose and tile_pos in [(0, 0), (0, 1), (1, 1), (1, 6)]:
                    self.save_rois = True
                else:
                    self.save_rois = False
 
                tile_bar = tqdm(
                    total=extra_iterations + 2,
                    desc=f"Tile {tile_pos} Optimization",
                    unit="iter",
                    leave=False,
                    bar_format="{l_bar}{bar} [iter: {n_fmt}/{total_fmt}{postfix}]",
                    position=1
                )

                best_mean_shift = np.inf
                best_roi_int = 0
                best_grad = np.zeros(2)

                # Optimise for the gradient using the chosen method
                iteration = 0 
                while iteration <= (extra_iterations + 1):
                    if iteration == 0:
                        grad = np.zeros(2)  # Start with zero gradient
                        shift = _get_shift(grad, tile_pos)

                        if method_type == MethodType.BROYSDEN:
                            # Initialize an approximation of the inverse Jacobian
                            J_inv = np.eye(2) * self.scaling_const 
                    else:
                        if method_type == MethodType.BROYSDEN:
                            # Compute new gradient estimate
                            grad_new = grad - J_inv @ shift
                            shift_new = _get_shift(grad_new, tile_pos)

                            # Compute Broyden’s update for J_inv
                            delta_grad = grad_new - grad
                            delta_shift = shift_new - shift

                            denom = np.dot(delta_shift, delta_shift)
                            if denom > 1e-8:  # Prevent division instability
                                J_inv += np.outer(delta_grad - J_inv @ delta_shift, delta_shift) / denom

                            # Update values for the next iteration
                            grad, shift = grad_new, shift_new

                        elif method_type == MethodType.SECANT:
                            if iteration == 1:
                                # Get the first estimate of the gradient
                                grad_new = -shift * self.scaling_const
                            else:
                                # Calculate the new gradient using the secant method
                                grad_new = grad - damping * shift * (grad - grad_prev) / (shift - shift_prev)

                            shift_new = _get_shift(grad_new, tile_pos)

                            grad_prev, grad = grad, grad_new
                            shift_prev, shift = shift, shift_new

                        else:
                            # Step the gradient in the direction of the shift
                            # This method depends on the orientation of the camera, and it may fail if the camera is rotated
                            grad -= shift * self.scaling_const
                            shift = _get_shift(grad, tile_pos)

                    # Calculate the error
                    mean_shift = np.sqrt(np.sum(shift ** 2))
                    tile_bar.set_postfix({"shift": f"{[round(s, 3) for s in shift.tolist()]}", 
                                        "grad": f"{[round(g, 4) for g in grad.tolist()]}", 
                                        "mean shift": f"{mean_shift:.3f}"})
                    
                    # Update best known solution
                    if mean_shift < best_mean_shift:
                        best_mean_shift = mean_shift
                        best_roi_int = self.roi.mean_intensity
                        best_grad = grad.copy()

                    if mean_shift < 0.1:
                        # Break if the shift is smaller than 0.1 pixels
                        break  # Converged

                    iteration += 1
                    tile_bar.update(1)
 
                # Store the intensity of the tile
                self.tile_ints[i, j] = best_roi_int
 
                # Store the gradient of the tile
                self.gradients[i, j, :] = best_grad

                print(mean_shift)
 
                tqdm_bar.set_postfix({"tile intensity": f"{self.tile_ints[i, j]:.6f}"})
 
                tqdm_bar.update(1)

                num_steps_final[i, j] = min(iteration, extra_iterations + 1)

        num_steps_mean = np.mean(num_steps_final)
        print(f"Mean number of steps: {num_steps_mean:.2f}")
 
        # Analyse the amplitude information
        self.amp_fit = AmpFit(self.tile_size, self.tile_ints)
        self.amp_wav = self.amp_fit.fit_amplitude()
 
        # Analyse the phase information
        if phase_approx_method == PhaseApproxMethod.GLIP:
            self.phase_fit = PhaseFitGLIP(self.tile_size, self.gradients)
        elif phase_approx_method == PhaseApproxMethod.MESH:
            self.phase_fit = PhaseFitMesh(self.tile_size, self.gradients)
        else:
            self.phase_fit = PhaseFitPoisson(self.tile_size, self.gradients)
        self.phase_wav = self.phase_fit.fit_phase()
 
    def get_whole_tilt_pattern(self):
        pattern = np.zeros(self.slm_shape, dtype=np.ushort)
 
        for i in range(self.tile_positions[0]):
            for j in range(self.tile_positions[1]):
                tile_pos = (i, j)
                grad_x, grad_y = self.gradients[i, j, :]
                tile = self._get_tilted_tile(grad_x, grad_y)
                self._place_tile(tile, tile_pos, pattern)
 
        return pattern
 
    def _get_tilted_tile(self, grad_x, grad_y):
        # Get the zernike tilt phase
        x, y = np.linspace(0, self.tile_size, self.tile_size, endpoint=False), np.linspace(0, self.tile_size, self.tile_size, endpoint=False)
        m, n = np.meshgrid(y, x, indexing='ij')
        tile = grad_y * m + grad_x * n # m is the y-axis, n is the x-axis
 
        return tile
 
    def _place_tile(self, tile, tile_pos=None, pattern=None):
        if pattern is None:
            pattern = np.zeros(self.slm_shape, dtype=np.ushort)
 
        if tile_pos is None:
            # Place the tile at the centre
            tile_M = tile_N = self.tile_size
            c_m, c_n = self.slm_shape[1] // 2, self.slm_shape[0] // 2
            pattern[c_n - tile_N//2:c_n + tile_N//2, c_m - tile_M//2:c_m + tile_M//2] = self._quantise_phase(tile)
        else:
            tile_M = tile_N = self.tile_size
            tile_m, tile_n = tile_pos
            pattern[tile_m * tile_M:(tile_m+1) * tile_M, tile_n * tile_N:(tile_n+1) * tile_N] = self._quantise_phase(tile)
        
        return pattern
 
    def _quantise_phase(self, phase):
        phase_wrapped = np.mod(phase, 2 * np.pi)
        grayscale_values = np.round(phase_wrapped / (2 * np.pi) * self.grayscale_range).astype(np.ushort) % self.grayscale_range
        return grayscale_values
 
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
 
    def save(self, exp_io: ExpIO):
        exp_io.save_npy(self.tile_ints, "amp_wavefront_raw_intensity")
        exp_io.save_npy(self.gradients, "tile_gradients")
        exp_io.save_npy(self.amp_wav, "amp_wavefront")
        exp_io.save_npy(self.phase_wav, "phase_wavefront")
 
    def load(self, exp_io: ExpIO, path):
        self.tile_ints = exp_io.load_npy(path + "amp_wavefront_raw_intensity.npy")
        pass
    
    def plot_gradients(self):
        return self.phase_fit.plot_gradients()
    
    def plot_phase(self):
        return self.phase_fit.plot_phase()
    
    def plot_amp(self):
        return self.amp_fit.plot_amp()

class AmpFit:
    def __init__(self, tile_size, measured_intensities):
        self.tile_size = tile_size
        self.tile_ints = measured_intensities
        self.M, self.N = self.tile_ints.shape
 
        # Declare the variables
        self.tile_amps = None
        self.amp_fitted = None

    def fit_amplitude(self):
        norm_intensity = self.tile_ints / np.max(self.tile_ints)
        self.tile_amps = np.sqrt(norm_intensity) 

        # Interpolate tile_amps to amp_fitted using cubic interpolation
        # tile_amps has size of M x N, but each tile has implicit size of tile_size x tile_size
        # amp_fitted should have size self.M * self.tile_size x self.N * self.tile_size
        x = np.linspace(0, self.N, self.N)
        y = np.linspace(0, self.M, self.M)
        x_new = np.linspace(0, self.N, self.N * self.tile_size)
        y_new = np.linspace(0, self.M, self.M * self.tile_size)
        X, Y = np.meshgrid(x, y)
        X_new, Y_new = np.meshgrid(x_new, y_new)
        self.amp_fitted = griddata((X.flatten(), Y.flatten()), self.tile_amps.flatten(), (X_new, Y_new), method='cubic')
        
        return self.amp_fitted
    
    def plot_amp(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot tile amplitudes
        im1 = axes[0].imshow(self.tile_amps, cmap='hot')
        axes[0].set_title('Tile Amplitudes')
        axes[0].set_xlabel('Tile X')
        axes[0].set_ylabel('Tile Y')
        # Add colorbar for tile amplitudes
        divider1 = make_axes_locatable(axes[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)

        # Plot interpolated amplitude
        im2 = axes[1].imshow(self.amp_fitted, cmap='hot')
        axes[1].set_title('Interpolated Amplitude')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        # Add colorbar for interpolated amplitude
        divider2 = make_axes_locatable(axes[1])
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax2)

        plt.tight_layout()

        return fig
    
class PhaseFitGeneric:
    def __init__(self, tile_size, measured_gradients):
        self.tile_size = tile_size
        self.measured_gradients = measured_gradients
        self.M, self.N, _ = measured_gradients.shape
        self.phase_fitted = None # Fitted phase

    def fit_phase(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def get_tile_gradients(self, phase):
        gradients_recovered = np.zeros((self.M, self.N, 2))
        for i in range(self.M):
            for j in range(self.N):
                tile = phase[i*self.tile_size:(i+1)*self.tile_size, j*self.tile_size:(j+1)*self.tile_size]
                tile = np.unwrap(tile, axis=0)
                tile = np.unwrap(tile, axis=1)
                gradients_recovered[i, j, 0] = np.mean(np.diff(tile, axis=1)) # x gradient
                gradients_recovered[i, j, 1] = np.mean(np.diff(tile, axis=0)) # y gradient

        return gradients_recovered # Radians per pixel
    
    def get_fitted_gradients(self):
        return self.get_tile_gradients(self.phase_fitted)
    
    def plot_gradients(self):
        recovered_gradients = self.get_tile_gradients(self.phase_fitted)
 
        fig, ax = plt.subplots(figsize=(6, 5))
        scale = 5
 
        # Plot measured gradients
        ax.quiver(
            np.arange(self.N), np.arange(self.M),
            self.measured_gradients[:, :, 0] * self.tile_size, self.measured_gradients[:, :, 1] * self.tile_size,
            color='b', scale=scale, scale_units='xy', label='Measured Gradients'
        )
 
        # Plot recovered gradients
        ax.quiver(
            np.arange(self.N), np.arange(self.M),
            recovered_gradients[:, :, 0] * self.tile_size, recovered_gradients[:, :, 1] * self.tile_size,
            color='r', scale=scale, scale_units='xy', label='Recovered Gradients'
        )
 
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-0.5, self.N - 0.5)
        ax.set_ylim(self.M - 0.5, -0.5)
        ax.set_xlabel('Tile Index X')
        ax.set_ylabel('Tile Index Y')
        ax.set_title('Gradients Comparison')
        ax.legend()
 
        plt.tight_layout()
 
        return fig
    
    def plot_phase(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
 
        # Plot fitted phase
        im = axes[0].imshow(self.phase_fitted, cmap='gray')
        axes[0].set_title('Fitted Phase')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        # Add colorbar for the fitted phase
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
 
        # Plot fitted phase (wrapped)
        phase_fitted_wrapped = self.phase_fitted % (2 * np.pi)
        im = axes[1].imshow(phase_fitted_wrapped, cmap='gray')
        axes[1].set_title('Fitted Phase (Wrapped)')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
 
        # Add colorbar for the wrapped phase
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
 
        plt.tight_layout()
 
        return fig

    
class PhaseFitPoisson(PhaseFitGeneric):
    def __init__(self, tile_size, measured_gradients):
        super().__init__(tile_size, measured_gradients)
 
    def fit_phase(self):
        # Extract the gradients from the tiles
        grad_x = self.measured_gradients[:, :, 0]
        grad_y = self.measured_gradients[:, :, 1]

        # Define the grid for interpolation
        x = np.linspace(0, self.N - 1, self.N)
        y = np.linspace(0, self.M - 1, self.M)
        xx, yy = np.meshgrid(x, y)

        # Flatten the grid and data for griddata
        points = np.array([xx.ravel(), yy.ravel()]).T
        grad_x_flat = grad_x.ravel()
        grad_y_flat = grad_y.ravel()

        # Define the new grid for interpolation
        x_new = np.linspace(0, self.N - 1, self.N * self.tile_size)
        y_new = np.linspace(0, self.M - 1, self.M * self.tile_size)
        xx_new, yy_new = np.meshgrid(x_new, y_new)

        # Perform cubic interpolation on the gradients
        grad_x_interpolated = griddata(points, grad_x_flat, (xx_new, yy_new), method='cubic') / self.tile_size
        grad_y_interpolated = griddata(points, grad_y_flat, (xx_new, yy_new), method='cubic') / self.tile_size

        # Integrate gradients to get the phase map
        self.phase_fitted = self.poisson_reconstruct(grad_x_interpolated, grad_y_interpolated)

        return self.phase_fitted

    def poisson_reconstruct(self, grad_x, grad_y):
        """
        Reconstruct a phase map from its gradients using the Poisson equation.
        
        Parameters:
            grad_x (numpy.ndarray): Gradient along x-axis
            grad_y (numpy.ndarray): Gradient along y-axis
        
        Returns:
            numpy.ndarray: Reconstructed phase map
        """
        h, w = grad_x.shape

        # Compute divergence
        div = np.zeros((h, w))
        div[:-1, :] += grad_y[:-1, :]
        div[1:, :]  -= grad_y[:-1, :]
        div[:, :-1] += grad_x[:, :-1]
        div[:, 1:]  -= grad_x[:, :-1]

        # Apply Discrete Cosine Transform (DCT)
        dct_div = dct(dct(div.T, norm='ortho').T, norm='ortho')

        # Create frequency grid
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        denom = (2 * np.cos(np.pi * xx / w) - 2) + (2 * np.cos(np.pi * yy / h) - 2)
        denom[0, 0] = 1  # Avoid division by zero at (0,0)

        # Solve Poisson equation in frequency domain
        dct_h = dct_div / denom
        dct_h[0, 0] = 0  # Set the mean to 0 to avoid drift

        # Inverse DCT to get the phase map
        phase = idct(idct(dct_h.T, norm='ortho').T, norm='ortho')

        # Normalize the phase
        phase -= phase.min()

        return phase

class PhaseFitMesh(PhaseFitGeneric):
    def __init__(self, tile_size, measured_gradients):
        super().__init__(tile_size, measured_gradients)

        self.z_optim = None # Optimized z coordinates of the vertices

    def fit_phase(self):
        z_init = np.ones((self.M+1, self.N+1))
        self.mse_loss_history = []  # Initialize an empty list to store the MSE cost history

        def callback(z_flat):
            mse = self._mse_loss(z_flat, self.measured_gradients)
            self.mse_loss_history.append(mse)  # Append the current MSE to the history

        result = minimize(
            self._mse_loss,
            z_init.flatten(),
            args=(self.measured_gradients,),
            method='L-BFGS-B',
            callback=callback
        )
        self.z_optim = result.x.reshape((self.M + 1, self.N + 1))
        self.phase_fitted = self._get_interp_z(self.z_optim, self.tile_size)
        return self.phase_fitted

    def _mse_loss(self, z_flat, gradients):
        z = z_flat.reshape((self.M + 1, self.N + 1))
        computed_gradients = self._compute_tile_gradients(z)
        grad_diff = computed_gradients - gradients
        mse = np.mean(grad_diff ** 2)
        return mse
    
    def _extract_tile_z(self, z, m, n):
        zA = z[m, n]   # A (top-left)
        zB = z[m, n+1] # B (top-right)
        zC = z[m+1, n+1] # C (bottom-right)
        zD = z[m+1, n] # D (bottom-left)
        zE = (zA + zB + zC + zD) / 4  # Central vertex E
        return zA, zB, zC, zD, zE

    def _compute_tile_gradients(self, z):
        M, N = z.shape[0] - 1, z.shape[1] - 1
        gradients = np.zeros((M, N, 2))
        for m in range(M):
            for n in range(N):
                zA, zB, zC, zD, _ = self._extract_tile_z(z, m, n)

                # Compute gradients as the mean of the gradients at the four triangles
                dz_dx = -0.5*zA + 0.5*zB + 0.5*zC - 0.5*zD
                dz_dy = -0.5*zA - 0.5*zB + 0.5*zC + 0.5*zD

                # Store the gradients
                gradients[m, n, 0] = dz_dx
                gradients[m, n, 1] = dz_dy
        return gradients

    def _get_interp_z(self, z, sampling):
        M, N = z.shape[0] - 1, z.shape[1] - 1
        interp_z = np.zeros((M * sampling, N * sampling))
        for m in range(M):
            for n in range(N):
                zA, zB, zC, zD, zE = self._extract_tile_z(z, m, n)
                for i in range(sampling):
                    for j in range(sampling):
                        y = i / (sampling - 1)
                        x = j / (sampling - 1)
                        if x >= y and y < 1 - x:  # Triangle ABE
                            mAB = 0.5 * (zB + zA)
                            u = x
                            v = y * 2
                            interp_z[m * sampling + i, n * sampling + j] = zA + u * (zB - zA) + v * (zE - mAB)
                        elif y >= 1 - x and y <= x:  # Triangle BCE
                            mBC = 0.5 * (zC + zB)
                            u = y
                            v = (1 - x) * 2
                            interp_z[m * sampling + i, n * sampling + j] = zB + u * (zC - zB) + v * (zE - mBC)
                        elif x < y and x >= 1 - y:  # Triangle CDE
                            mCD = 0.5 * (zD + zC)
                            u = (1 - x)
                            v = (1 - y) * 2
                            interp_z[m * sampling + i, n * sampling + j] = zC + u * (zD - zC) + v * (zE - mCD)
                        else:  # Triangle DAE
                            mDA = 0.5 * (zA + zD)
                            u = (1 - y)
                            v = x * 2
                            interp_z[m * sampling + i, n * sampling + j] = zD + u * (zA - zD) + v * (zE - mDA)
        interp_z = interp_z - np.min(interp_z)
        return interp_z
    
    def plot_loss(self):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(self.mse_loss_history)
        ax.set_yscale('log')  # Set y-axis to log scale
        ax.set_title('MSE Loss History')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('MSE Loss (log scale)')

        plt.tight_layout()
        
        return fig

class PhaseFitGLIP(PhaseFitGeneric):
    def __init__(self, tile_size, measured_gradients):
        super().__init__(tile_size, measured_gradients)

        # Optimized gradients of the tiles
        self.grax_x_optim = None
        self.grax_y_optim = None

    def fit_phase(self):
        # Reconstruct the gradient surface from the measured gradients
        grad_x = self.measured_gradients[:, :, 0]
        grad_y = self.measured_gradients[:, :, 1]

        self.grad_x_optim = self.reconstruct_surface(grad_x, self.tile_size)
        self.grad_y_optim = self.reconstruct_surface(grad_y, self.tile_size)

        # Integrate gradients to get the phase map
        self.phase_fitted = self.poisson_reconstruct(self.grad_x_optim, self.grad_y_optim)

        return self.phase_fitted

    def reconstruct_surface(self, A, tile_size):
        """
        Reconstructs a continuous surface B of size (M*tile_size, N*tile_size)
        from an input array A of shape (M, N), such that the mean over each
        tile_size x tile_size block of B equals the corresponding value in A.

        Uses piecewise bilinear patches. Solves L v = b in least-squares sense
        to handle the underdetermined system.

        Parameters
        ----------
        A : 2D numpy array of shape (M, N)
            Input array of block means.
        tile_size : int
            Size `a` of each square patch in the reconstructed surface.

        Returns
        -------
        B : 2D numpy array of shape (M*tile_size, N*tile_size)
            Reconstructed smooth surface.
        """
        M, N = A.shape
        a = tile_size
        num_cells = M * N
        num_corners = (M+1) * (N+1)

        # Build sparse system L v = b
        rows, cols, data = [], [], []
        b = np.zeros(num_cells)
        eq = 0
        for i in range(M):
            for j in range(N):
                idxs = [i*(N+1)+j, (i+1)*(N+1)+j, i*(N+1)+(j+1), (i+1)*(N+1)+(j+1)]
                for idx in idxs:
                    rows.append(eq)
                    cols.append(idx)
                    data.append(1.0)
                b[eq] = 4.0 * A[i, j]
                eq += 1
        L = sp.coo_matrix((data, (rows, cols)), shape=(num_cells, num_corners)).tocsr()

        # Solve for corner values v
        v, *_ = spla.lsqr(L, b)

        V = v.reshape((M+1, N+1))

        # Reconstruct B via bilinear interpolation
        H, W = M*a, N*a
        B = np.zeros((H, W))
        xs = np.linspace(0, 1, a, endpoint=False)[:, None]
        ys = np.linspace(0, 1, a, endpoint=False)[None, :]
        for i in range(M):
            for j in range(N):
                v00 = V[i, j]
                v10 = V[i+1, j]
                v01 = V[i, j+1]
                v11 = V[i+1, j+1]
                patch = (v00*(1-xs)*(1-ys) + v10*xs*(1-ys)
                        + v01*(1-xs)*ys + v11*xs*ys)
                B[i*a:(i+1)*a, j*a:(j+1)*a] = patch
        return B

    def poisson_reconstruct(self, grad_x, grad_y):
        """
        Reconstruct a phase map from its gradients using the Poisson equation
        with Neumann boundary conditions, implemented via even reflection and FFT.

        Parameters:
            grad_x (numpy.ndarray): Gradient along x-axis
            grad_y (numpy.ndarray): Gradient along y-axis

        Returns:
            numpy.ndarray: Reconstructed phase map
        """
        h, w = grad_x.shape

        # Compute divergence
        div = np.zeros((h, w))
        div[:-1, :] += grad_y[:-1, :]
        div[1:, :]  -= grad_y[:-1, :]
        div[:, :-1] += grad_x[:, :-1]
        div[:, 1:]  -= grad_x[:, :-1]

        # Build even extension by reflection
        H, W = 2*h, 2*w

        div_ext = np.zeros((H, W))
        div_ext[:h, :w] = div
        div_ext[h:, :w] = div[::-1, :]
        div_ext[:h, w:] = div[:, ::-1]
        div_ext[h:, w:] = div[::-1, ::-1]

        # FFT of the divergence
        F_div = np.fft.fft2(div_ext)

        # Frequency grid
        ky = np.fft.fftfreq(H)[:, None]
        kx = np.fft.fftfreq(W)[None, :]
        denom = (2*np.pi*kx)**2 + (2*np.pi*ky)**2
        denom[0, 0] = 1.0  # avoid div by 0

        # Solve Poisson equation in frequency domain
        F_phi = F_div / denom
        F_phi[0, 0] = 0  # set mean to 0

        # Inverse FFT
        phi_ext = np.fft.ifft2(F_phi).real

        # Crop back to original size and take the negative sign
        phase = -phi_ext[:h, :w]

        # Normalize
        phase -= phase.min()

        return phase

    # # Alternative Poisson reconstruction using DCT - it is equivalent to the one above (commented out)
    # def poisson_reconstruct(self, grad_x, grad_y):
    #     """
    #     Reconstruct a phase map from its gradients using the Poisson equation.
        
    #     Parameters:
    #         grad_x (numpy.ndarray): Gradient along x-axis
    #         grad_y (numpy.ndarray): Gradient along y-axis
        
    #     Returns:
    #         numpy.ndarray: Reconstructed phase map
    #     """
    #     h, w = grad_x.shape

    #     # Compute divergence
    #     div = np.zeros((h, w))
    #     div[:-1, :] += grad_y[:-1, :]
    #     div[1:, :]  -= grad_y[:-1, :]
    #     div[:, :-1] += grad_x[:, :-1]
    #     div[:, 1:]  -= grad_x[:, :-1]

    #     # Apply Discrete Cosine Transform (DCT)
    #     dct_div = dct(dct(div.T, norm='ortho').T, norm='ortho')

    #     # Create frequency grid
    #     x = np.arange(w)
    #     y = np.arange(h)
    #     xx, yy = np.meshgrid(x, y)
    #     denom = (2 * np.cos(np.pi * xx / w) - 2) + (2 * np.cos(np.pi * yy / h) - 2)
    #     denom[0, 0] = 1  # Avoid division by zero at (0,0)

    #     # Solve Poisson equation in frequency domain
    #     dct_h = dct_div / denom
    #     dct_h[0, 0] = 0  # Set the mean to 0

    #     # Inverse DCT to get the phase map
    #     phase = idct(idct(dct_h.T, norm='ortho').T, norm='ortho')

    #     # Normalize the phase
    #     phase -= phase.min()

    #     return phase