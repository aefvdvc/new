import numpy as np


data = np.load('./data/density_est_20250308_123514_indep_gmm_x_dim=2_y_dim=2_alpha=10.0_beta=10.0/py_est_at_epoch35.npy')


print(data)

# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]


print("Shape:", data.shape)
print("Data type:", data.dtype)