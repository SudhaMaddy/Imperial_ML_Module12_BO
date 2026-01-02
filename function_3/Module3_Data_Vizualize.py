import numpy as np
import matplotlib.pyplot as plt

# Load data
X = np.load("/home/sudhakarmadamala/Downloads/Initial_data_points_starter/initial_data/function_3/initial_inputs.npy")
print(X)
y = np.load("/home/sudhakarmadamala/Downloads/Initial_data_points_starter/initial_data/function_3/initial_outputs.npy")
print(y)
# Plot x1 vs x2
print(" ------------------- x1 vs x2 --------------------------")
plt.scatter(X[:,0], X[:,1], c=y, cmap="viridis")
plt.colorbar(label="Function output")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Function 3 – x1 vs x2")
plt.show()

print(" ------------------- x1 vs x3 --------------------------")
# x1 vs x3
plt.scatter(X[:,0], X[:,2], c=y, cmap="viridis")
plt.colorbar(label="Function output")
plt.xlabel("x1")
plt.ylabel("x3")
plt.title("Function 3 – x1 vs x3")
plt.show()
print(" ------------------- x2 vs x3 --------------------------")

# x2 vs x3
plt.scatter(X[:,1], X[:,2], c=y, cmap="viridis")
plt.colorbar(label="Function output")
plt.xlabel("x2")
plt.ylabel("x3")
plt.title("Function 3 – x2 vs x3")
plt.show()
print(" ------------------- 3d --------------------------")

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap='viridis')
fig.colorbar(sc, label="Function output")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.set_title("Function 3 – 3D view")

plt.show()

