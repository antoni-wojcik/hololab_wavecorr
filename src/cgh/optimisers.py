import torch
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod

from src.cgh.propagator import IPropagator

# -----------------------------
# Phase Optimisation Interface
# -----------------------------

class IPhaseOptim(ABC):
    def __init__(self):
        self.loss_history = None
        pass

    @abstractmethod
    def optim(self, phase_init_np):
        pass

# -----------------------------
# Phase Optimisation Functions
# -----------------------------

# Batch Gradient Descent-based phase optimization using PyTorch autograd for gradient computation. It optimizes the phase pattern to minimize the NMSE between the target intensity and the intensity obtained from the far-field of the propagator.
class PhaseOptimBGD(IPhaseOptim):
    def __init__(self, prop: IPropagator, target_int, num_iter=100, batch_size=10, precalc_near_field=True):
        """
        Arguments:
        prop: the propagator to use for computing the far-field and intensity.
        target_int: the target intensity pattern to optimize towards.
        num_iter: number of iterations for the optimization.
        batch_size: the batch size for the optimization. The phase pattern will be optimized in batches of this size, which can help with convergence and memory usage.
        precalc_near_field: if True, computes the near-field here instead of inside the propagator.
        """
        self.prop = prop
        self.set_target(target_int)
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.precalc_near_field = precalc_near_field
        self.loss_history = np.zeros(num_iter)

    def set_target(self, target_int):
        self.target_int = torch.tensor(target_int, dtype=torch.float32, device=self.prop.device)

    def _loss(self, far_field, target_int):
        """Computes the NMSE loss between the target intensity and the intensity obtained from the far-field of the propagator."""
        approx_int = self.prop.get_intensity(far_field)
        nmse = torch.mean((approx_int / torch.mean(approx_int) - target_int / torch.mean(target_int)) ** 2)

        return nmse

    def optim(self, phase_init_np, learning_rate=0.01):
        # Initialize phase tensor
        phase = torch.tensor(phase_init_np, dtype=torch.float32, requires_grad=True, device=self.prop.device)

        phase = torch.nn.Parameter(phase)

        optimizer = torch.optim.Adam([phase], lr=learning_rate)

        pbar = tqdm(range(self.num_iter), desc="Batch Gradient Descent", unit="iter")
        for i in pbar:
            optimizer.zero_grad()
            
            if self.precalc_near_field:
                # Compute the near-field once and reuse it
                field_near = torch.exp(1j * phase)
                field_far = self.prop.get_far_field(field_near)
            else:
                field_far = self.prop.get_far_field(phase)
            
            # Compute the loss
            current_loss = self._loss(field_far, self.target_int)
            current_loss.backward()
            
            optimizer.step()

            with torch.no_grad():
                phase.copy_(phase % (2 * np.pi))
            
            self.loss_history[i] = current_loss.item()
            pbar.set_postfix(loss=f"{current_loss.item():.6f}")

        # Store the final far-field
        self.field_far = field_far.clone().detach()

        return phase.detach().cpu().numpy()
