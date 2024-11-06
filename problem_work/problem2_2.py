"""Ising problem question."""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import linear_model

# Define lattice size
l = 40  # noqa: E741

# Create 10000 random Ising states
rng = (
    np.random.default_rng()
)  # You can also specify a seed here if needed, e.g., np.random.default_rng(seed=42)

# Generate 10000 random Ising states using the generator
states = rng.choice([-1, 1], size=(10000, l))

# %%


def ising_energies(states: np.array, l: int) -> np.array:  # noqa: E741
    """This function calculates the energies of the states in the nearest-neighbor Ising.

    Args:
        states (np.array): Our states
        l (int): Lattice Size

    Returns:
        np.array: Our energy values for the states
    """
    j = np.zeros(
        (l, l),
    )
    for i in range(l):
        j[i, (i + 1) % l] -= 1.0
    # Compute energies
    return np.einsum("...i,ij,...j->...", states, j, states)


# Calculate Ising energies
energies = ising_energies(states, l)

# %%

# Reshape Ising states into RL samples: S_iS_j --> X_p
states = np.einsum("...i,...j->...ij", states, states)
shape = states.shape
states = states.reshape((shape[0], shape[1] * shape[2]))

# Build final data set
Data = [states, energies]

# Define number of samples
n_samples = 500

# Define train and test data sets
X_train = Data[0][:n_samples]
Y_train = Data[1][:n_samples]
X_test = Data[0][n_samples : 3 * n_samples // 2]
Y_test = Data[1][n_samples : 3 * n_samples // 2]

# %%

# Set up Lasso and Ridge Regression models
leastsq = linear_model.LinearRegression()
ridge = linear_model.Ridge()
lasso = linear_model.Lasso()

# Define error lists
train_errors_leastsq, test_errors_leastsq = [], []
train_errors_ridge, test_errors_ridge = [], []
train_errors_lasso, test_errors_lasso = [], []

# Set regularization strength values
lmbdas = np.logspace(-4, 5, 10)

# Initialize coefficients for Ridge regression and Lasso
coefs_leastsq, coefs_ridge, coefs_lasso = [], [], []

# %%

for lmbda in lmbdas:
    # Ordinary least squares
    leastsq.fit(X_train, Y_train)
    coefs_leastsq.append(leastsq.coef_)
    train_errors_leastsq.append(leastsq.score(X_train, Y_train))
    test_errors_leastsq.append(leastsq.score(X_test, Y_test))

    # Ridge regression
    ridge.set_params(alpha=lmbda)
    ridge.fit(X_train, Y_train)
    coefs_ridge.append(ridge.coef_)
    train_errors_ridge.append(ridge.score(X_train, Y_train))
    test_errors_ridge.append(ridge.score(X_test, Y_test))

    # Lasso regression
    lasso.set_params(alpha=lmbda)
    lasso.fit(X_train, Y_train)
    coefs_lasso.append(lasso.coef_)
    train_errors_lasso.append(lasso.score(X_train, Y_train))
    test_errors_lasso.append(lasso.score(X_test, Y_test))

    # Plot Ising interaction J for each model
    J_leastsq = np.array(leastsq.coef_).reshape((l, l))
    J_ridge = np.array(ridge.coef_).reshape((l, l))
    J_lasso = np.array(lasso.coef_).reshape((l, l))

    cmap_args = {"vmin": -1.0, "vmax": 1.0, "cmap": "seismic"}

    fig, axarr = plt.subplots(nrows=1, ncols=3)

    axarr[0].imshow(J_leastsq, **cmap_args)
    axarr[0].set_title("$\\mathrm{OLS}$", fontsize=16)
    axarr[0].tick_params(labelsize=16)

    axarr[1].imshow(J_ridge, **cmap_args)
    axarr[1].set_title(f"$\\mathrm{{Ridge}},\\ \\lambda={lmbda:.4f}$", fontsize=16)
    axarr[1].tick_params(labelsize=16)

    im = axarr[2].imshow(J_lasso, **cmap_args)
    axarr[2].set_title(f"$\\mathrm{{LASSO}},\\ \\lambda={lmbda:.4f}$", fontsize=16)
    axarr[2].tick_params(labelsize=16)

    divider = make_axes_locatable(axarr[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)

    cbar.ax.set_yticklabels(np.arange(-1.0, 1.0 + 0.25, 0.25), fontsize=14)
    cbar.set_label("$J_{i,j}$", labelpad=-40, y=1.12, fontsize=16, rotation=0)

    fig.subplots_adjust(right=2.0)

    plt.show()

# %%

# Plot performance on both the training and test data
plt.semilogx(lmbdas, train_errors_leastsq, "b", label="Train (OLS)")
plt.semilogx(lmbdas, test_errors_leastsq, "--b", label="Test (OLS)")
plt.semilogx(lmbdas, train_errors_ridge, "r", label="Train (Ridge)", linewidth=1)
plt.semilogx(lmbdas, test_errors_ridge, "--r", label="Test (Ridge)", linewidth=1)
plt.semilogx(lmbdas, train_errors_lasso, "g", label="Train (LASSO)")
plt.semilogx(lmbdas, test_errors_lasso, "--g", label="Test (LASSO)")

fig = plt.gcf()
fig.set_size_inches(10.0, 6.0)

plt.legend(loc="lower left", fontsize=16)
plt.ylim([-0.01, 1.01])
plt.xlim([min(lmbdas), max(lmbdas)])
plt.xlabel(r"$\lambda$", fontsize=16)
plt.ylabel("Performance $R^2$", fontsize=16)
plt.tick_params(labelsize=16)
plt.show()
# %%
