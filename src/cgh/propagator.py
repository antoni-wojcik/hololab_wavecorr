import torch
import numpy as np
import scipy.special

from abc import ABC, abstractmethod


class IPropagator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_far_field(self, near_field):
        pass

    @abstractmethod
    def get_near_field(self, far_field):
        pass

    @abstractmethod
    def get_intensity(self, field):
        pass


# -----------------------------------------------------
# Class for a propagator that account for position-independentaberrations.
# -----------------------------------------------------

class PropagatorTorch(IPropagator):
    def __init__(self, slm_shape, field_inc=None, scale = 2, square_far_field=False):
        """
        A propagator that uses PyTorch for the FFTs. It forces the far-field to have square aspect ratio. The far-field is upscaled by a factor of 2 in each dimension compared to the near-field to account for bandwidth. It applies a sinc envelope to the far-field to correct for the square pixels. It also operates on 3D tensors, where the first dimension is the batch size (K) - therefore, the near-field and far-field are of shape (K, P, Q) and (K, M, N) respectively.

        slm_shape: tuple of (P, Q) dimensions of the SLM.
        field_inc: optional incident field (P, Q) to be applied to the near-field, complex dtype.
        scale: integer factor to upscale the far-field. Default is 2.
        square_far_field: if True, forces far-field to be square-shaped.
        """
        self.P, self.Q = slm_shape
        self.square_far_field = square_far_field

        if self.square_far_field:
            self.M = self.N = scale * max(self.P, self.Q)
        else:
            # If not square, use the provided scale for each dimension
            self.M, self.N = scale * self.P, scale * self.Q

        # Set the device for PyTorch operations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create the coordinate arrays for the SLM and far-field
        self.Y, self.X = self._get_xy_grid()
        self.U, self.V = self._get_uv_grid()

        # Create the sinc envelope for the far-field propagation
        self.sinc_env = self._get_sinc_env()

        # Set the incident field
        self.set_field_inc(field_inc)
        
    def set_field_inc(self, field_inc):
        # Initialize the incident field if provided
        if field_inc is not None:
            self.field_inc = torch.tensor(field_inc, dtype=torch.complex128, device=self.device)
        else:
            self.field_inc = torch.ones((self.P, self.Q), dtype=torch.complex128, device=self.device)

        # Calculate energy normalization factor
        self.energy_norm = torch.sqrt(torch.mean(torch.abs(self.field_inc)**2))

    def _pad(self, field):
        """
        Pads the tensor field (shape (K, P, Q)) or (P, Q) to (K, M, N) or (M, N) with zeros,
        centering the original field in the new array.
        This works for both 2D and 3D tensors.
        """
        pad_top = (self.M - self.P) // 2
        pad_bottom = self.M - self.P - pad_top
        pad_left = (self.N - self.Q) // 2
        pad_right = self.N - self.Q - pad_left

        # torch.nn.functional.pad works for both 2D and 3D tensors when padding last two dims
        return torch.nn.functional.pad(field, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    def _crop(self, x):
        """
        Crops the array/tensor x to the center region of shape (P, Q).
        Supports both 2D (M, N) and 3D (K, M, N) tensors.
        """
        pad_top = (self.M - self.P) // 2
        pad_bottom = self.M - self.P - pad_top
        pad_left = (self.N - self.Q) // 2
        pad_right = self.N - self.Q - pad_left

        if x.ndim == 3:
            # x: (K, M, N)
            return x[:, pad_top:self.M - pad_bottom, pad_left:self.N - pad_right]
        elif x.ndim == 2:
            # x: (M, N)
            return x[pad_top:self.M - pad_bottom, pad_left:self.N - pad_right]
        else:
            raise ValueError("Input must be a 2D or 3D tensor.")

    def _get_xy_grid(self):
        p_indices = torch.fft.fftshift(torch.fft.fftfreq(self.P, d=1/self.P)).to(self.device)
        q_indices = torch.fft.fftshift(torch.fft.fftfreq(self.Q, d=1/self.Q)).to(self.device)
        Y, X = torch.meshgrid(p_indices, q_indices, indexing='ij')
        return Y, X

    def _get_uv_grid(self):
        u_indices = torch.fft.fftshift(torch.fft.fftfreq(self.M)).to(self.device)
        v_indices = torch.fft.fftshift(torch.fft.fftfreq(self.N)).to(self.device)
        U, V = torch.meshgrid(u_indices, v_indices, indexing='ij')
        return U, V
    
    def _get_sinc_env(self):
        sinc_env = torch.sinc(self.U) * torch.sinc(self.V)
        return sinc_env

    def get_far_field(self, field_near):
        # field_near: (K, P, Q)
        # field_inc: (P, Q)
        field = field_near * self.field_inc  # Broadcasting over batch
        field = self._pad(field)  # (K, M, N)
        # FFT on last two dims
        field = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm='ortho', dim=(-2, -1)),
            dim=(-2, -1)
        )
        field = field / self.energy_norm  # Normalize the energy of the field
        field_far = field * self.sinc_env  # (M, N) will broadcast to (K, M, N)
        return field_far
        
    def get_near_field(self, field_far):
        # field_far: (K, M, N)
        field = field_far / self.sinc_env  # (M, N) will broadcast to (K, M, N)
        field = torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm='ortho', dim=(-2, -1)),
            dim=(-2, -1)
        )
        field = self._crop(field)  # (K, P, Q)
        field_near = field / self.field_inc  # Broadcasting over batch
        return field_near
    
    def get_intensity(self, field):
        """
        Compute the intensity of the field.
        If field has 3 dimensions, take the mean across the first one after torch.abs.
        """
        intensity = torch.abs(field) ** 2
        if intensity.ndim == 3:
            intensity = intensity.mean(dim=0)
        return intensity
    
    def phase_to_near(self, phase):
        """
        Compute the near-field from a phase (numpy or torch tensor).
        The phase is assumed to be in radians.
        """
        phase_tensor = torch.tensor(phase, dtype=torch.float64, device=self.device)
        field_near = torch.exp(1j * phase_tensor)
        return field_near
    
    def get_slm_shape(self):
        """
        Returns the shape of the SLM.
        """
        return (self.P, self.Q)
    
    def get_far_shape(self):
        """
        Returns the shape of the far-field.
        """
        return (self.M, self.N)

# -----------------------------------------------------
# Class for Seidel aberration coefficients and propagators that accounts for them
# -----------------------------------------------------

class AberrCoefficients:
    def __init__(self, coma=0.0, astigmatism=0.0, field_curvature=0.0, distortion=0.0, tilt_x=0.0, tilt_y=0.0):
        """
        Container class for Aberration coefficients.
        coma: Coefficient for Seidel coma aberration.
        astigmatism: Coefficient for Seidel astigmatism aberration.
        field_curvature: Coefficient for Seidel field curvature aberration.
        distortion: Coefficient for Seidel distortion aberration.
        tilt_x: Coefficient for tilt in the x direction.
        tilt_y: Coefficient for tilt in the y direction.
        """
        self.coma = coma
        self.astigmatism = astigmatism
        self.field_curvature = field_curvature
        self.distortion = distortion
        self.tilt_x = tilt_x
        self.tilt_y = tilt_y

        # Store the coefficient names in a list for easy access
        self.coeff_names = ['Coma', 'Astigmatism', 'Field Curvature', 'Distortion', 'Tilt X', 'Tilt Y']

class SeidelPropagatorSeries(IPropagator):
    def __init__(self, slm_shape, aberr: AberrCoefficients, field_inc=None, scale = 2, square_far_field=False, num_terms=None, precompute=True, high_precision=True, chebyshev_approx=False):
        """
        A propagator that accounts for Seidel aberrations using the series expansion-based approach.
        It forces the far-field to have square aspect ratio. The far-field is upscaled by a factor of 2 in each dimension compared to the near-field to account for bandwidth. It applies a sinc envelope to the far-field to correct for the square pixels. It also operates on 3D tensors, where the first dimension is the batch size (K) - therefore, the near-field and far-field are of shape (K, P, Q) and (K, M, N) respectively.

        slm_shape: tuple of (P, Q) dimensions of the SLM.
        field_inc: optional incident field (P, Q) to be applied to the near-field, complex dtype.
        scale: integer factor to upscale the far-field. Default is 2.
        square_far_field: if True, forces far-field to be square-shaped.
        curvature: Seidel curvature coefficient for the wavefront.
        tilt: Tilt coefficient for the wavefront, important if the screen is tilted with respect to the optical axis.
        num_terms: number of terms in the series expansion for the Seidel aberrations.
        precompute: if True, precomputes the factors for the series expansion to speed up the computation, at a cost of memory.
        high_precision: if True, uses high precision for the calculations (float64, complex128), otherwise uses (float32, complex64).
        chebyshev_approx: if True, uses Chebyshev expansion, which can be more efficient and accurate for large m than the Taylor expansion. It is only applied if precompute is True.

        """
        self.P, self.Q = slm_shape

        self.square_far_field = square_far_field
        if self.square_far_field:
            self.M = self.N = scale * max(self.P, self.Q)
        else:
            # If not square, use the provided scale for each dimension
            self.M, self.N = scale * self.P, scale * self.Q

        # Set the device for PyTorch operations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if high_precision:
            # Use high precision for the calculations
            self.float_dtype = torch.float64
            self.complex_dtype = torch.complex128
        else:
            # Use standard precision for the calculations
            self.float_dtype = torch.float32
            self.complex_dtype = torch.complex64

        # Create the coordinate arrays for the SLM and far-field
        self.Y, self.X = self._get_xy_grid()
        self.V, self.U = self._get_uv_grid()

        # Create the sinc envelope for the far-field propagation
        self.sinc_env = self._get_sinc_env()

        # Set the incident field
        self.set_field_inc(field_inc)

        # Get normalized coordinates for Seidel aberrations. They are defined so that the radius <= 1.0. 
        r = np.sqrt(self.P**2 + self.Q**2)
        Xn = self.X * 2.0 / r
        Yn = self.Y * 2.0 / r
        Un = self.U * np.sqrt(2.0)
        Vn = self.V * np.sqrt(2.0)
        R2n = Xn**2 + Yn**2  # Normalized radius squared
        H2n = Un**2 + Vn**2  # Normalized frequency squared

        self.num_terms = num_terms # Number of terms in the series expansion

        k_base = aberr.field_curvature * H2n + aberr.tilt_x * Un + aberr.tilt_y * Vn
        r_base = R2n

        print(f"Max k_base: {k_base.abs().max():.3e}, Max r_base: {r_base.abs().max():.3e}")

        self.chebyshev_approx = chebyshev_approx
        self.precompute = precompute or chebyshev_approx  # Chebyshev approximation requires precomputation of the polynomials and coefficients, so we set precompute to True if chebyshev_approx is True.

        # Initialize the r and k factors
        self.cplx_units = torch.tensor([1, 1j, -1, -1j], dtype=self.complex_dtype, device=self.device)  # 1, i, -1, -i

        if self.precompute:
            self._precompute_factors(r_base, k_base)
        else:
            eps = 1e-12
            self.r_log = torch.log(torch.clamp(r_base.abs(), min=eps)).double()
            self.k_log = torch.log(torch.clamp(k_base.abs(), min=eps)).double()
            self.k_sign = torch.stack([torch.ones_like(k_base), torch.sign(k_base)], dim=0)

            self.fact_log = torch.zeros(self.num_terms, dtype=self.float_dtype, device=self.device)
            for m in range(1, self.num_terms):
                self.fact_log[m] = self.fact_log[m-1] + torch.log(torch.tensor(m, dtype=self.float_dtype, device=self.device))

    def _precompute_factors(self, r_base, k_base):
        if not self.chebyshev_approx:
            # Do Taylor expansion
            # precompute log(|f|)
            eps = 1e-12
            log_k_abs = torch.log(torch.clamp(k_base.abs(), min=eps)).double()

            # precompute log-factorials (small, CPU or GPU)
            log_fact = torch.zeros(self.num_terms, dtype=self.float_dtype, device=self.device)
            for m in range(1, self.num_terms):
                log_fact[m] = log_fact[m-1] + torch.log(torch.tensor(m, dtype=self.float_dtype, device=self.device))

            self.r_factors = torch.zeros((self.num_terms, self.P, self.Q), dtype=self.float_dtype, device=self.device)  # (num, P, Q)
            self.k_phase = torch.zeros((self.num_terms, self.M, self.N), dtype=torch.float32, device=self.device)  # (num, M, N)
            self.k_log_scale = torch.zeros((self.num_terms, self.M, self.N), dtype=self.float_dtype, device=self.device)  # (num, M, N)

            r_pow = torch.ones((self.P, self.Q), dtype=self.float_dtype, device=self.device)
            k_phase = torch.ones((self.M, self.N), dtype=torch.float32, device=self.device)
            for m in range(self.num_terms):
                # store
                self.k_phase[m] = k_phase

                # log-scale for this term: m*log|f| - log(m!)
                log_scale = m * log_k_abs - log_fact[m]
                self.k_log_scale[m] = log_scale

                # update phase only: (sign(f))^m
                k_phase = k_phase * torch.sign(k_base)

                # update r_pow for the next term
                self.r_factors[m] = r_pow
                
                r_pow = r_pow * r_base

        else:
            # Do Chebyshev expansion
            self.T_polys = torch.zeros((self.num_terms, self.P, self.Q), dtype=self.float_dtype, device=self.device)  # (num, M, N)
            
            self.T_polys[0] = torch.ones_like(r_base)
            self.T_polys[1] = r_base
            for m in range(2, self.num_terms):
                self.T_polys[m] = 2 * r_base * self.T_polys[m-1] - self.T_polys[m-2]

            self.coeffs = torch.empty((self.num_terms, self.M, self.N), dtype=self.complex_dtype, device=self.device)
            for m in range(self.num_terms):
                J_m = torch.from_numpy(scipy.special.jv(m, k_base.cpu().numpy())).to(self.float_dtype).to(self.device)

                if m == 0:
                    self.coeffs[m] = J_m
                else:
                    self.coeffs[m] = 2 * J_m * self.cplx_units[m % 4]  # The (1j)**(m) factor comes from the series expansion of the exponential of the cosine term in the Chebyshev expansion
        
        
    def set_field_inc(self, field_inc):
        # Initialize the incident field if provided
        if field_inc is not None:
            self.field_inc = torch.tensor(field_inc, dtype=self.complex_dtype, device=self.device)
        else:
            self.field_inc = torch.ones((self.P, self.Q), dtype=self.complex_dtype, device=self.device)

        # Calculate energy normalization factor
        self.energy_norm = torch.sqrt(torch.mean(torch.abs(self.field_inc)**2))

    def _pad(self, field):
        """
        Pads the tensor field (shape (K, P, Q)) or (P, Q) to (K, M, N) or (M, N) with zeros,
        centering the original field in the new array.
        This works for both 2D and 3D tensors.
        """
        pad_top = (self.M - self.P) // 2
        pad_bottom = self.M - self.P - pad_top
        pad_left = (self.N - self.Q) // 2
        pad_right = self.N - self.Q - pad_left

        # torch.nn.functional.pad works for both 2D and 3D tensors when padding last two dims
        return torch.nn.functional.pad(field, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    def _get_xy_grid(self):
        p_indices = torch.fft.fftshift(torch.fft.fftfreq(self.P, d=1/self.P)).to(self.device)
        q_indices = torch.fft.fftshift(torch.fft.fftfreq(self.Q, d=1/self.Q)).to(self.device)
        Y, X = torch.meshgrid(p_indices, q_indices, indexing='ij')
        return Y, X

    def _get_uv_grid(self):
        m_indices = torch.fft.fftshift(torch.fft.fftfreq(self.M)).to(self.device)
        n_indices = torch.fft.fftshift(torch.fft.fftfreq(self.N)).to(self.device)
        V, U = torch.meshgrid(m_indices, n_indices, indexing='ij')
        return V, U
    
    def _get_sinc_env(self):
        sinc_env = torch.sinc(self.U) * torch.sinc(self.V)
        return sinc_env

    def _fft2(self, field_near):
        """
        Perform a 2D FFT on the near-field tensor, which is expected to be of shape (K, P, Q).
        The output will be of shape (K, M, N) after padding and FFT.
        """
        # field_near: (K, P, Q)
        field_pad = self._pad(field_near)  # (K, M, N)
        # FFT on last two dims
        field_far = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(field_pad, dim=(-2, -1)), norm='ortho', dim=(-2, -1)),
            dim=(-2, -1)
        )
        return field_far
    
    
    def get_far_field(self, field_near):
        """
        Compute the far-field from the near-field using the series expansion method.
        field_near: (K, P, Q) tensor representing the near-field.
        Returns:
        field_far: (K, M, N) tensor representing the far-field.
        """
        
        # Initialize field_far with the same batch dimension as field_near (supports 2D or 3D input)
        out_shape = (field_near.shape[0], self.M, self.N) if field_near.ndim == 3 else (self.M, self.N)
        field_far = torch.zeros(out_shape, dtype=self.complex_dtype, device=self.device)
        
        # Add the incident field to the near-field
        field_temp = field_near * self.field_inc

        # Series expansion for the far-field
        # If precompute is True, use the precomputed r and k factors
        if self.precompute:
            if not self.chebyshev_approx:
                max_log = self.k_log_scale.max()  # Get the maximum log scale for numerical stability

                for m in range(self.num_terms):
                    # apply r^m
                    field_near_m = self.r_factors[m] * field_temp

                    # FFT
                    field_far_m = self._fft2(field_near_m)

                    # compute scale = exp(log_scale)
                    log_scale = self.k_log_scale[m]

                    scale = torch.exp(log_scale - max_log)  

                    phase = self.cplx_units[m % 4] * self.k_phase[m]

                    # scaled accumulation
                    field_far.add_(scale * (phase * field_far_m))

                field_far = torch.exp(max_log) * field_far  # Rescale the final result by the maximum log scale to restore the correct magnitude

            else:
                for m in range(self.num_terms):
                    field_near_m = self.T_polys[m] * field_temp
                    field_far_m = self._fft2(field_near_m)
                    field_far.add_(self.coeffs[m] * field_far_m)
        else:
            # # Iterate over the number of terms in the series expansion using the base factors directly, without precomputation. Use exponential of logs to maintain numerical stability and increase speed.
            
            for m in range(self.num_terms):
                # Apply the r factor to the near-field
                field_near_m = torch.exp(self.r_log * m) * field_temp
                
                # Compute the far-field for this term
                field_far_m = self._fft2(field_near_m)
                
                # Update the far-field with the k factor
                field_far.add_(self.cplx_units[m % 4] * torch.exp(self.k_log * m - self.fact_log[m]) * self.k_sign[m % 2] * field_far_m)
        
        # Normalize the energy of the field and apply the sinc envelope
        field_far = field_far / self.energy_norm
        field_far = field_far * self.sinc_env
        return field_far
    
    def get_near_field(self, far_field):
        """
        Near-field is not defined for this propagator. Calculating the inverse is a difficult problem.
        """
        pass
    
    
    def get_intensity(self, field):
        """
        Compute the intensity of the field.
        If field has 3 dimensions, take the mean across the first one after torch.abs.
        """
        intensity = field.real**2 + field.imag**2  # Compute intensity
        if intensity.ndim == 3:
            intensity = intensity.mean(dim=0)
        return intensity
    
    def phase_to_near(self, phase):
        """
        Compute the near-field from a phase (numpy or torch tensor).
        The phase is assumed to be in radians.
        """
        phase_tensor = torch.tensor(phase, dtype=self.float_dtype, device=self.device)
        field_near = torch.exp(1j * phase_tensor)
        return field_near
    
    def get_slm_shape(self):
        """
        Returns the shape of the SLM.
        """
        return (self.P, self.Q)
    
    def get_far_shape(self):
        """
        Returns the shape of the far-field.
        """
        return (self.M, self.N)
     
class SeidelPropagatorPatches(IPropagator):
    def __init__(self, slm_shape, aberr: AberrCoefficients, field_inc=None, scale = 2, square_far_field=False, num_tiles=20, high_precision=False, bound=10, use_blending=False, vectorise=False):
        """
        A propagator that accounts for Seidel aberrations using the patch-based approach. 
        The far-field is upscaled by a factor of scale in each dimension compared to the near-field to account for bandwidth. It applies a sinc envelope to the far-field to correct for the square pixels. It also operates on 3D tensors, where the first dimension is the batch size (K) - therefore, the near-field and far-field are of shape (K, P, Q) and (K, M, N) respectively.
        slm_shape: tuple of (P, Q) dimensions of the SLM.
        field_inc: optional incident field (P, Q) to be applied to the near-field, complex dtype.
        scale: integer factor to upscale the far-field. Default is 2.
        square_far_field: if True, forces far-field to be square-shaped.
        curvature: Seidel curvature coefficient for the wavefront.
        tilt: Tilt coefficient for the wavefront, important if the screen is tilted with respect to the optical axis.
        num_tiles: number of tiles in the far-field. The far-field is divided into num_tiles x num_tiles tiles, each of size (M // num_tiles, N // num_tiles).
        high_precision: if True, uses high precision for the calculations (float64, complex128), otherwise uses (float32, complex64).
        bound: additional bound to the tile size to account for the boundary effects and blending.
        use_blending: if True, uses a blending window to smooth the edges of the tiles. However, there is a weird behaviour at the edges of the tiles caused by convolution with the PSF, so I found it better to just crop them. It is best to set boundary to a small integer value, like 10 and disable blending. It is only applied if bound > 0.
        vectorise: if True, uses vectorized operations for computing the far-field. This is faster, but requires more memory, and may run slower if not enough memory is available.
        """
        self.P, self.Q = slm_shape

        self.square_far_field = square_far_field
        if self.square_far_field:
            self.M = self.N = scale * max(self.P, self.Q)
        else:
            # If not square, use the provided scale for each dimension
            self.M, self.N = scale * self.P, scale * self.Q

        # Set the device for PyTorch operations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if high_precision:
            # Use high precision for the calculations
            self.float_dtype = torch.float64
            self.complex_dtype = torch.complex128
        else:
            # Use standard precision for the calculations
            self.float_dtype = torch.float32
            self.complex_dtype = torch.complex64

        # Create the coordinate arrays for the SLM and far-field
        self.Y, self.X = self._get_xy_grid()
        self.V, self.U = self._get_uv_grid()

        # Create the sinc envelope for the far-field propagation
        self.sinc_env = self._get_sinc_env()

        # Set the incident field
        self.set_field_inc(field_inc)

        # Prepare the dimensions
        self.num_tiles = num_tiles # Number of terms in the series expansion
        self.tile_dims = np.array([self.M // self.num_tiles, self.N // self.num_tiles])  # Dimensions of each tile in the far-field.
        self.bound = bound # Additional bound to the tiles to account for the boundary effects and blending.

        # Precompute the OTFs for the tiles
        self.aberr = aberr
        self.otf_array = None 
        self._precompute_otfs()

        self.vectorise = vectorise  # If True, uses vectorized operations for computing the far-field.
        if self.vectorise:
            self.fold = torch.nn.Fold(
                output_size=(self.M, self.N),
                kernel_size=(self.tile_dims[0], self.tile_dims[1]),
                stride=(self.tile_dims[0], self.tile_dims[1]),
            )

        # Define the blending window for the tiles
        self.use_blending = use_blending
        
        if self.bound > 0 and self.use_blending:
            # If the boundary is greater than 0, compute the blending window
            self.blending_window = self._compute_blending_window()

            # Create the fold operation for blending
            self.fold_overlap = torch.nn.Fold(
                output_size=(self.M + 2 * self.bound, self.N + 2 * self.bound),
                kernel_size=(self.tile_dims[0] + 2 * self.bound, self.tile_dims[1] + 2 * self.bound),
                stride=(self.tile_dims[0], self.tile_dims[1]),
            )

            # Create the tensor to normalize the effects of blending
            dummy = torch.ones(self.num_tiles, self.num_tiles, self.tile_dims[0] + 2 * self.bound, self.tile_dims[1] + 2 * self.bound, dtype=self.float_dtype, device=self.device)
            self.blend_weight = self._tile_overlapping(dummy * self.blending_window)
            self.blend_weight = 1.0 / (self.blend_weight + 1e-6)

    def _compute_blending_window(self):
        """
        Computes a 2D blending window for the tiles, which ramps up and down at the edges.
        The window is of size (tile_dims_b[0], tile_dims_b[1]) and has a linear ramp from 0 to 1 at the edges.
        """
        def _ramp_up_down(l, b):
            idx = torch.arange(l, dtype=self.float_dtype, device=self.device)
            w1d = torch.ones(l, dtype=self.float_dtype, device=self.device)
            if b > 0:
                # Linear ramp-up on left
                left = idx[:b]
                ramp_up = (left + 1) / b      # values from 1/b ... 1
                w1d[:b] = ramp_up.clamp(0.0, 1.0)
                # Linear ramp-down on right
                # ramp_down at position L-b+i: (b - i)/b, for i=0..b-1
                ramp_down = (b - torch.arange(1, b+1, device=self.device, dtype=self.float_dtype)) / b
                w1d[l - b:l] = ramp_down.clamp(0.0, 1.0)

            return w1d
        
        tile_dims_b = self.tile_dims + 2 * self.bound  # Size of the tile in the far-field after adding the boundary.

        w1d_y = _ramp_up_down(tile_dims_b[0], self.bound)
        w1d_x = _ramp_up_down(tile_dims_b[1], self.bound)

        w2d = w1d_y[:, None] * w1d_x[None, :]  # Outer product to create 2D window
        return w2d

    def _get_aberr_phase(self, X, Y, u0, v0):
        # For now only calculate the wavefront curvature and tilt.
        beta = torch.atan2(v0, u0)
        h2 = u0**2 + v0**2
        h = torch.sqrt(h2)  # Normalized radius
        R2 = X**2 + Y**2  # Normalized radius squared
        Xr = X * torch.cos(beta) + Y * torch.sin(beta)

        phase = torch.zeros_like(R2)

        if self.aberr.coma != 0:
            phase += self.aberr.coma * h * R2 * Xr
        if self.aberr.astigmatism != 0:
            phase += self.aberr.astigmatism * h2 * Xr**2
        if self.aberr.field_curvature != 0 or self.aberr.tilt_x != 0 or self.aberr.tilt_y != 0:
            phase += (self.aberr.field_curvature * h2 + self.aberr.tilt_x * u0 + self.aberr.tilt_y * v0) * R2
        if self.aberr.distortion != 0:
            phase += self.aberr.distortion * h2 * h * Xr

        return phase

    def _precompute_otfs(self):
        # Get normalized coordinates for Seidel aberrations. They are defined so that the radius <= 1.0. 
        r = np.sqrt(self.P**2 + self.Q**2)
        Xn = self.X * 2.0 / r
        Yn = self.Y * 2.0 / r
        un = self.U[0] * np.sqrt(2.0) # Take the first row of U
        vn = self.V[:,0] * np.sqrt(2.0) # Take the first column of V

        # Calculate important dimensions
        tile_dims_b = self.tile_dims + 2 * self.bound  # Size of the tile in the far-field after adding a boundary.

        # Calculate the otf for each tile
        psf_tiles = torch.zeros((self.num_tiles, self.num_tiles, tile_dims_b[0] * 2, tile_dims_b[1] * 2), dtype=self.complex_dtype, device=self.device)
        otf_tiles = torch.zeros((self.num_tiles, self.num_tiles, tile_dims_b[0] * 2, tile_dims_b[1] * 2), dtype=self.complex_dtype, device=self.device)

        for tile_r in range(self.num_tiles):
            for tile_s in range(self.num_tiles):
                # Center of the tile in normalized coordinates
                v0 = vn[tile_r * self.tile_dims[0] + self.tile_dims[0] // 2]
                u0 = un[tile_s * self.tile_dims[1] + self.tile_dims[1] // 2]

                # Calculate the aberration phase for this tile
                aberr_phase = self._get_aberr_phase(Xn, Yn, u0, v0)

                # Create the near-field for this tile
                field_near_tile = torch.exp(1j * aberr_phase)

                # Take the FFT to get the PSF for that tile
                field_far_tile = self._fft2(field_near_tile)

                # Pad the far-field to the size of the tile with bound
                field_far_tile = self._pad_bound(field_far_tile, 2 * self.bound)

                # Compute the far-field for this tile using FFT
                # Crop the center tile_size_pad region
                # This is the PSF of that tile
                center_m = field_far_tile.shape[-2] // 2
                center_n = field_far_tile.shape[-1] // 2
                half_size = tile_dims_b // 2
                psf = field_far_tile[
                    center_m - half_size[0] : center_m + half_size[0],
                    center_n - half_size[1] : center_n + half_size[1]
                ]
                
                # Pad the PSF to double the size
                psf = self._pad2(psf)

                # Compute the OTF of the tile
                otf = self._ifft2(psf)

                # Store the OTF for this tile
                psf_tiles[tile_r, tile_s] = psf
                otf_tiles[tile_r, tile_s] = otf
        
        # Store all the OTFs
        self.psf_array = psf_tiles
        self.otf_array = otf_tiles
        
        
    def set_field_inc(self, field_inc):
        # Initialize the incident field if provided
        if field_inc is not None:
            self.field_inc = torch.tensor(field_inc, dtype=self.complex_dtype, device=self.device)
        else:
            self.field_inc = torch.ones((self.P, self.Q), dtype=self.complex_dtype, device=self.device)

        # Calculate energy normalization factor
        self.energy_norm = torch.sqrt(torch.mean(torch.abs(self.field_inc)**2))

    def _pad(self, field):
        """
        Pads the tensor field (shape (K, P, Q)) or (P, Q) to (K, M, N) or (M, N) with zeros,
        centering the original field in the new array.
        This works for both 2D and 3D tensors.
        """
        pad_top = (self.M - self.P) // 2
        pad_bottom = self.M - self.P - pad_top
        pad_left = (self.N - self.Q) // 2
        pad_right = self.N - self.Q - pad_left

        # torch.nn.functional.pad works for both 2D and 3D tensors when padding last two dims
        return torch.nn.functional.pad(field, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    def _pad2(self, field):
        """
        Pad the tensor to double the size
        """
        pad_top = field.shape[-2] // 2
        pad_bottom = field.shape[-2] - pad_top
        pad_left = field.shape[-1] // 2
        pad_right = field.shape[-1] - pad_left

        return torch.nn.functional.pad(field, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    def _pad_bound(self, field, bound):
        """
        Pads the tensor field (shape (K, P, Q)) or (P, Q) by adding a circular boundary of width self.bound
        around each 2D “slice,” centering the original field in the new array.

        For a 2D tensor of shape (P, Q), returns shape (P + 2*bound, Q + 2*bound).
        For a 3D tensor of shape (K, P, Q), returns shape (K, P + 2*bound, Q + 2*bound).

        PYTORCH IS BROKEN AND IT DOES NOT WORK WITH 2D TENSORS, SO WE ADD A DUMMY BATCH DIMENSION.
        """

        if field.ndim == 2:
            # field: (P, Q) -> temporarily make it (1, P, Q) and squeeze back later
            return torch.nn.functional.pad(field.unsqueeze(0), (bound, bound, bound, bound), mode='circular').squeeze(0)

        elif field.ndim == 3:
            # pad each channel’s 2D slice
            return torch.nn.functional.pad(field, (bound, bound, bound, bound), mode='circular')

        else:
            raise ValueError(f"Unsupported field.ndim={field.ndim} in _pad_bound; expected 2 or 3.")
    
    def _crop2(self, field):
        center_m = field.shape[-2] // 2
        center_n = field.shape[-1] // 2
        half_m = field.shape[-2] // 4
        half_n = field.shape[-1] // 4
        return field[..., 
            center_m - half_m : center_m + half_m,
            center_n - half_n : center_n + half_n
        ]

    def _get_xy_grid(self):
        p_indices = torch.fft.fftshift(torch.fft.fftfreq(self.P, d=1/self.P)).to(self.device)
        q_indices = torch.fft.fftshift(torch.fft.fftfreq(self.Q, d=1/self.Q)).to(self.device)
        Y, X = torch.meshgrid(p_indices, q_indices, indexing='ij')
        return Y, X

    def _get_uv_grid(self):
        m_indices = torch.fft.fftshift(torch.fft.fftfreq(self.M)).to(self.device)
        n_indices = torch.fft.fftshift(torch.fft.fftfreq(self.N)).to(self.device)
        V, U = torch.meshgrid(m_indices, n_indices, indexing='ij')
        return V, U
    
    def _get_sinc_env(self):
        sinc_env = torch.sinc(self.U) * torch.sinc(self.V)
        return sinc_env

    def _fft2(self, field_near, pad=True):
        """
        Perform a 2D FFT on the near-field tensor, which is expected to be of shape (K, P, Q).
        The output will be of shape (K, M, N) after padding and FFT.
        """
        # field_near: (K, P, Q)
        if pad:
            field_pad = self._pad(field_near)  # (K, M, N)
        else:
            field_pad = field_near

        # FFT on last two dims
        field_far = torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.ifftshift(field_pad, dim=(-2, -1)), norm='ortho', dim=(-2, -1)
            ), dim=(-2, -1)
        )
        return field_far
    
    def _ifft2(self, field_far):
        field_near = torch.fft.ifftshift(
            torch.fft.ifft2(
                torch.fft.fftshift(field_far, dim=(-2, -1)), norm='ortho', dim=(-2, -1)
            ), dim=(-2, -1)
        )
        return field_near
    
    def _tile_overlapping(self, tiles):
        """
        Stitch the tiles together with overlapping regions.
        tiles: (K, num_tiles, num_tiles, tile_dims[0] + 2 * bound, tile_dims[1] + 2 * bound)
        Returns:
        field_overlap: (K, M, N)
        """
        tile_res = tiles.reshape(-1, self.num_tiles * self.num_tiles,
                                 (self.tile_dims[0] + 2 * self.bound) * (self.tile_dims[1] + 2 * self.bound)).permute(0, 2, 1)  # (K, (tile_dims[0] + 2 * bound) * (tile_dims[1] + 2 * bound), num_tiles * num_tiles)
        field_overlap = self.fold_overlap(tile_res).squeeze()
        field_overlap = field_overlap[..., self.bound:-self.bound, self.bound:-self.bound]
        return field_overlap
    
    def _tile(self, tiles):
        """
        Stitch the tiles together without overlapping regions.
        tiles: (K, num_tiles, num_tiles, tile_dims[0], tile_dims[1])
        Returns:
        field: (K, M, N)
        """
        tile_res = tiles.reshape(-1, self.num_tiles * self.num_tiles,
                                 self.tile_dims[0] * self.tile_dims[1]).permute(0, 2, 1)
                
        return self.fold(tile_res).squeeze()
    
    def get_far_field(self, field_near):
        """
        Compute the far-field from the near-field using the tile-based method.
        This method applies the incident field, computes the ideal far-field, and then convolves it with the PSF for each tile to account for the aberrations.
        Parameters:
        field_near: (K, P, Q) tensor representing the near-field.
        Returns:
        field_far: (K, M, N) tensor representing the far-field.
        """
        
        # Add the incident field to the near-field
        field_temp = field_near * self.field_inc

        # Get the ideal far-field without any aberrations
        field_far_ideal = self._fft2(field_temp)

        # Pad the ideal far-field with a boundary to account for the convolution with the PSF
        field_far_ideal_b = self._pad_bound(field_far_ideal, self.bound)

        # If vectorization is used, we can process all tiles at once
        if self.vectorise:
            # Divide the far-field into tiles (shape (K, num_tiles, num_tiles, tile_dims[0] + 2 * bound, tile_dims[1] + 2 * bound))
            tiles = field_far_ideal_b.unfold(-2, self.tile_dims[0] + 2 * self.bound, self.tile_dims[0]).unfold(-2, self.tile_dims[1] + 2 * self.bound, self.tile_dims[1])

            # Convolve the tiles with the PSF -> multiply its IFFT with OTF and FFT back
            tiles_pad = self._pad2(tiles)
            tiles_ifft = self._ifft2(tiles_pad)
            tiles_ifft_conv = tiles_ifft * self.otf_array
            tiles_conv = self._fft2(tiles_ifft_conv, pad=False)
            tiles_conv_crop = self._crop2(tiles_conv)

            # Stitch the far-field together from tiles experiencing different aberrations
            if not self.use_blending or self.bound == 0:
                if self.bound > 0:
                    # If bound is greater than 0, crop the tiles to the size of the tile_size
                    tiles_conv_crop = tiles_conv_crop[..., self.bound:-self.bound, self.bound:-self.bound]

                # Place the tiles in the far-field
                field_far = self._tile(tiles_conv_crop)
            else:
                # If blending is used, apply the blending window to the tiles and then stitch them together
                field_far_bound = self._tile_overlapping(tiles_conv_crop * self.blending_window)
                        
                # Normalize the errors from blending
                field_far = field_far_bound * self.blend_weight
        else:
            # If vectorization is not used, iterate over the tiles
            field_far = torch.zeros_like(field_far_ideal)

            if self.bound > 0 and self.use_blending:
                field_far_bound = torch.nn.functional.pad(field_far_ideal, (self.bound, self.bound, self.bound, self.bound), mode='constant', value=0.0)
            
            for tile_r in range(self.num_tiles):
                for tile_s in range(self.num_tiles):
                    tile = field_far_ideal_b[..., 
                        tile_r * self.tile_dims[0]:(tile_r + 1) * self.tile_dims[0] + 2 * self.bound,
                        tile_s * self.tile_dims[1]:(tile_s + 1) * self.tile_dims[1] + 2 * self.bound
                    ]
                    # Convolve the tile with the PSF -> multiply its IFFT with OTF and FFT back
                    tile_pad = self._pad2(tile)
                    tile_ifft = self._ifft2(tile_pad)
                    tile_ifft_conv = tile_ifft * self.otf_array[tile_r, tile_s]
                    tile_conv = self._fft2(tile_ifft_conv, pad=False)
                    tile_conv_crop = self._crop2(tile_conv)

                    # Place the tile
                    if self.bound == 0 or not self.use_blending:
                        if self.bound > 0:
                            # If bound is greater than 0, crop the tile to the size of the tile_size
                            tile_conv_crop = tile_conv_crop[..., self.bound:-self.bound, self.bound:-self.bound]
                        # Place the tile
                        field_far[..., 
                                  tile_r * self.tile_dims[0]:(tile_r + 1) * self.tile_dims[0],
                                  tile_s * self.tile_dims[1]:(tile_s + 1) * self.tile_dims[1]] = tile_conv_crop
                    else:
                        # Crop to size tile_size + 2 * bound and multiply by the blending window
                        field_far_bound[...,
                                tile_r * self.tile_dims[0]:(tile_r + 1) * self.tile_dims[0] + 2 * self.bound,
                                tile_s * self.tile_dims[1]:(tile_s + 1) * self.tile_dims[1] + 2 * self.bound] += tile_conv_crop * self.blending_window
                            
            if self.bound > 0 and self.use_blending:
                # Crop the field_far_bound to the original size
                field_far = field_far_bound[..., self.bound:-self.bound, self.bound:-self.bound]
                # Normalize the field_far_bound by the weight_sum
                field_far = field_far * self.blend_weight

        # Normalize the energy of the field and apply the sinc envelope
        field_far = field_far / self.energy_norm
        field_far = field_far * self.sinc_env
        return field_far
    
    def get_near_field(self, far_field):
        """
        Near-field is not defined for this propagator. Calculating the inverse is a difficult problem.
        """
        pass
    
    
    def get_intensity(self, field):
        """
        Compute the intensity of the field.
        If field has 3 dimensions, take the mean across the first one after torch.abs.
        """
        intensity = field.real**2 + field.imag**2  # Compute intensity
        if intensity.ndim == 3:
            intensity = intensity.mean(dim=0)
        return intensity
    
    def phase_to_near(self, phase):
        """
        Compute the near-field from a phase (numpy or torch tensor).
        The phase is assumed to be in radians.
        """
        phase_tensor = torch.tensor(phase, dtype=self.float_dtype, device=self.device)
        field_near = torch.exp(1j * phase_tensor)
        return field_near
    
    def get_slm_shape(self):
        """
        Returns the shape of the SLM.
        """
        return (self.P, self.Q)
    
    def get_far_shape(self):
        """
        Returns the shape of the far-field.
        """
        return (self.M, self.N)
