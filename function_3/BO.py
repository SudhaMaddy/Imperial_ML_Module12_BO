import numpy as np
import matplotlib.pyplot as plt
#%pip install scikit-learn 


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C



# Load data
X = np.load("/home/sudhakarmadamala/Downloads/Initial_data_points_starter/initial_data/function_4/initial_inputs.npy")
print(X)
y = np.load("/home/sudhakarmadamala/Downloads/Initial_data_points_starter/initial_data/function_4/initial_outputs.npy")
print(y)
# Plot x1 vs x2

# ---- YOUR chosen point
x_user = np.array([0.520000, 0.650000, 0.300000])

kernel = C(1.0) * RBF(length_scale=0.5)

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,
    normalize_y=True
)

gp.fit(X, y)

n_candidates = 5000
X_candidates = np.random.rand(n_candidates, 3)
#----------

beta = 2.0  # exploration strength

mu, sigma = gp.predict(X_candidates, return_std=True)

ucb = mu + beta * sigma


#------ next point to submit
best_idx = np.argmax(ucb)
x_next = X_candidates[best_idx]

print("Next suggested point (x1, x2, x3):")
print(x_next)
#------
formatted = "-".join([f"{v:.6f}" for v in x_next])
print("Submit this:")
print(formatted)
#See the my choice and BO generated values
# Plot BOTH points for comparison
plt.figure()
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', s=60)

plt.scatter(x_user[0], x_user[1],
            color='red', s=150, marker='X', label='My choice')

plt.scatter(x_next[0], x_next[1],
            color='blue', s=150, marker='D', label='BO suggestion')

plt.colorbar(label="Function output")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Function 3: My choice vs BO suggestion")
plt.legend()
plt.show()




#--------------------------------------------------------
plt.figure()
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', s=60)
plt.scatter(x_user[0], x_user[1],
            color='red', s=150, marker='X', label='My choice')
plt.colorbar(label="Function output")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Function 3: x1 vs x2")
plt.legend()
plt.show()
#--------------------------------------------------------
plt.figure()
plt.scatter(X[:,0], X[:,2], c=y, cmap='viridis', s=60)
plt.scatter(x_user[0], x_user[2],
            color='red', s=150, marker='X', label='My choice')
plt.colorbar(label="Function output")
plt.xlabel("x1")
plt.ylabel("x3")
plt.title("Function 3: x1 vs x3")
plt.legend()
plt.show()
#--------------------------------------------------------
plt.figure()
plt.scatter(X[:,1], X[:,2], c=y, cmap='viridis', s=60)
plt.scatter(x_user[1], x_user[2],
            color='red', s=150, marker='X', label='My choice')
plt.colorbar(label="Function output")
plt.xlabel("x2")
plt.ylabel("x3")
plt.title("Function 3: x2 vs x3")
plt.legend()
plt.show()

#--------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap='viridis', s=60)
ax.scatter(x_user[0], x_user[1], x_user[2],
           color='red', s=200, marker='X', label='My choice')

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.set_title("Function 3 – 3D view")
fig.colorbar(sc, label="Function output")
ax.legend()

plt.show()

#--------------------------------------------------------




