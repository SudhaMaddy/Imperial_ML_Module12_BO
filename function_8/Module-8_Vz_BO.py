import numpy as np
import matplotlib.pyplot as plt
import itertools
import math

# Load data
X = np.load("/home/sudhakarmadamala/Downloads/Initial_data_points_starter/initial_data/function_8/initial_inputs.npy")
print(X)
y = np.load("/home/sudhakarmadamala/Downloads/Initial_data_points_starter/initial_data/function_8/initial_outputs.npy")
print(y)

def plot_pairwise_projections(X, y, function_name="Function"):
    n_dims = X.shape[1]
    pairs = list(itertools.combinations(range(n_dims), 2))

    for i, j in pairs:
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(
            X[:, i],
            X[:, j],
            c=y,
            cmap="viridis",
            s=70,
            edgecolor="k"
        )
        plt.colorbar(sc, label="Function output")
        plt.xlabel(f"x{i+1}")
        plt.ylabel(f"x{j+1}")
        plt.title(f"{function_name}: x{i+1} vs x{j+1}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        
plot_pairwise_projections(X, y, function_name="Function 8")

#------------------------------------------------------------------- BO Function ---------------------------------

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

dim = X.shape[1]  # number of dimensions (4 for Function 6)
x_user = np.array([0.341337,0.127752,0.148573,0.141133,0.887783,0.257133,0.067495,0.824468])


# -----------------------------
# GAUSSIAN PROCESS (BO MODEL)
# -----------------------------
kernel = C(1.0) * RBF(length_scale=0.5)

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-3,
    normalize_y=True
)

gp.fit(X, y)

# -----------------------------
# BAYESIAN OPTIMISATION STEP
# -----------------------------
n_candidates = 1200
X_candidates = np.random.rand(n_candidates, dim)

beta = 1.5 # exploration vs exploitation

mu, sigma = gp.predict(X_candidates, return_std=True)
ucb = mu + beta * sigma

best_idx = np.argmax(ucb)
x_bo = X_candidates[best_idx]

print("BO suggested point:")
print(x_bo)

print("Submit BO value:")
print("-".join([f"{v:.6f}" for v in x_bo]))

# -----------------------------
# PAIRWISE VISUALISATION
# -----------------------------
pairs = list(itertools.combinations(range(dim), 2))

for i, j in pairs:
    plt.figure(figsize=(6, 5))
    
    sc = plt.scatter(
        X[:, i], X[:, j],
        c=y, cmap="viridis", s=60
    )
    
    # USER POINT
    plt.scatter(
        x_user[i], x_user[j],
        color="red", marker="X", s=150, label="My choice"
    )
    
    # BO POINT
    plt.scatter(
        x_bo[i], x_bo[j],
        color="blue", marker="D", s=120, label="BO suggestion"
    )
    
    plt.colorbar(sc, label="Function output")
    plt.xlabel(f"x{i+1}")
    plt.ylabel(f"x{j+1}")
    plt.title(f"Function 4: x{i+1} vs x{j+1}")
    plt.legend()
    plt.show()


