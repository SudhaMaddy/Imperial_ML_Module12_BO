import numpy as np

# Load inputs (X)
X = np.load(
    "/home/sudhakarmadamala/Downloads/Initial_data_points_starter/initial_data/function_1/initial_inputs.npy"
)
print("Inputs: ",X)

# Load outputs (y)
y = np.load(
    "/home/sudhakarmadamala/Downloads/Initial_data_points_starter/initial_data/function_1/initial_outputs.npy"
)
print("Output: ",y)

import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')
plt.colorbar(label="Function output")
plt.xlabel("x1")
plt.xlabel("x2")
plt.title("Function 1 – Initial data")
plt.show()
