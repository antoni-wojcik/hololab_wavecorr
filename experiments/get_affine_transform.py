# Script to find the affine transfrom between the experimental and target images

from src.calibration.affine_transform import AffineTransform
import numpy as np
import matplotlib.pyplot as plt
from src.io.experiment import ExpIO


target_path = r"data\experiments\2025-05-08_wavefront_correction_test\f6cm_exact\target.npy"
exp_path = r"data\experiments\2025-05-08_wavefront_correction_test\f6cm_exact\capture_6.npy"

target_img = np.load(target_path)
exp_img = np.load(exp_path)

aft = AffineTransform()

aft.find(exp_img, target_img)

# Show the aligned image
aligned_img = aft.apply(exp_img, target_img.shape)
plt.imshow(aligned_img)
plt.show()

# Save the affine transform
exp = ExpIO(name="affine_transform")
aft.save(exp_io=exp, name="aft_matrix")