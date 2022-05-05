
import matplotlib.pyplot as plt
import numpy as np
import empymod
from discretize import TensorMesh
from wbi import wavelet_regularization as regularization
from scipy.optimize import minimize

# Do you want to compute the Jacobian via finite differences IN PARALLEL?
parallel = True

if parallel:
    from joblib import Parallel, delayed



## Defining the Model and Mapping

# Here we generate a synthetic model and a mappig which goes from the model
# space to the row space of our linear operator.

nParam = 50  # Number of model parameters

depth = np.linspace(0,20, nParam)
# A 1D mesh
mesh = TensorMesh([np.r_[np.diff(depth),1]])

# Creating the true model
true_model = np.ones(nParam)*0.01
true_model[depth > 6] = 0.15
true_model[depth > 11] = 0.35
true_model[depth > 15] = 0.1

# Plotting the true model
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.plot(depth, true_model, "b-")
ax.set_ylim([-2, 2])
plt.show()

## Defining the forward model

def forward(d, EC, s, freq):
    """
    :param d: depths of the interfaces (in metre), excluding +- infty
    :param EC: conductivity profile (in Siemens per metre)
    :param f: frequency of the dipole (Hz)
    :param s: intercoil distance (array)
    :param verb: set to 4 for debugging
    :return:
    """


    EC = np.r_[1e-8, EC]  # Adding the air-layer
    res = 1 / EC
    inpdat = {'src': [0, 0, 0, -0.10, 90], 'rec': [s, np.zeros(s.shape), 0, -0.10, 90],
              # LHS assenstelsel!
              'depth': d, 'freqtime': freq, 'aniso': np.ones(EC.size),
              'htarg': {'pts_per_dec': -1}, 'verb': 1,
               'mrec': True}
    return empymod.loop(**inpdat, res=res).imag

## Predict Synthetic Data

# Here, we use the true model to create synthetic data which we will subsequently
# invert.

# Standard deviation of Gaussian noise being added
std = 0.01
np.random.seed(42)
dclean = np.r_[forward(depth, true_model , np.arange(1,40), 6400)]
dobs = dclean + dclean*np.random.rand(dclean.size)*std
nd = dobs.size
W = 1/(std*dobs) # reciprocals of estimated noise


# Define the Inverse Problem
# The inverse problem is defined by 3 things:

#     1) Data Misfit: a measure of how well our recovered model explains the field data
#     2) Regularization: constraints placed on the recovered model and a priori information
#     3) Optimization: the numerical approach used to solve the inverse problem

# Define the data misfit. Here the data misfit is the L2 norm of the weighted
# residual between the observed data and the data predicted for a given model.
# Within the data misfit, the residual between predicted and observed data are
# normalized by the data's standard deviation.

def dmisfit(m):
    m = np.exp(m)
    dpred = forward(depth, m , np.arange(1,40), 6400)
    return 1 / nd * np.linalg.norm(W * (dpred - dobs)) ** 2

def dmisfit_deriv(m):
    """
    Derivative via finite differences
    """
    m = np.exp(m)
    dpred = forward(depth, m , np.arange(1,40), 6400)
    residual = dpred - dobs
    h = 1e-8

    def process(j):
        delh = np.zeros(m.size)
        delh[j] = h
        return ( forward(depth, m+delh , np.arange(1,40), 6400) - dpred) / h

    if parallel:
        results = Parallel(n_jobs=-2,)(
            delayed(process)(i) for i in range(m.size))
    else:
        results = [process(i) for i in range(m.size)]

    J = np.vstack(results)
    deriv = J @ (W**2 * residual)
    return 2 / nd * deriv


##
# Define the regularization (model objective function).

# Play here with the wav-parameter
# - db1 = blocky
# - db2, db3, db4 = rather sharp
# - db5+ = rather smooth

reg = regularization.WaveletRegularization1D(mesh, wav="db6")
beta = 5e4

def phi(m):
    return dmisfit(m) + beta * reg(np.exp(m))


def jac(m):
    deriv = dmisfit_deriv(m) + beta * reg.deriv(np.exp(m))
    return deriv * np.exp(m)

starting_model = np.random.rand(nParam) * np.log(0.1) # Note, we work in log-domain to ensure positive EC-values
x = minimize(phi, starting_model, jac=jac, method='L-BFGS-B', options={'maxiter':250} )


## Plotting Results

# Observed versus predicted data
fig, ax = plt.subplots(1, 2, figsize=(12 * 1.2, 4 * 1.2))
ax[0].plot(dobs, "b-")
ax[0].plot(forward(depth, np.exp(x.x) , np.arange(1,40), 6400), "r-")
ax[0].legend(("Observed Data", "Predicted Data"))

# True versus recovered model
ax[1].plot(mesh.vectorCCx, true_model, "b-")
ax[1].plot(mesh.vectorCCx, np.exp(x.x), "r-")
ax[1].legend(("True Model", "Recovered Model"))
# ax[1].set_ylim([-2, 2])
ax[1].set_title("Wavelet-type " + reg.wavelets.wav)

plt.show()

##
fig, ax_ls = plt.subplots(2,2)
wav_list = ['db1', 'db2', 'db3', 'db6']
betalist = [1e2, 5e1, 5e1, 1e4]
for idx, wav in enumerate(wav_list):
    reg = regularization.WaveletRegularization1D(mesh, wav=wav)
    beta = betalist[idx]
    def phi(m):
        return dmisfit(m) + beta * reg(np.exp(m))


    def jac(m):
        deriv = dmisfit_deriv(m) + beta * reg.deriv(np.exp(m))
        return deriv * np.exp(m)

    starting_model = np.random.rand(nParam) * np.log(0.1) # Note, we work in log-domain to ensure positive EC-values
    x = minimize(phi, starting_model, jac=jac, method='L-BFGS-B', options={'maxiter':250} )
    ax_ls[idx//2, idx%2].plot(mesh.vectorCCx, true_model, "b-")
    ax_ls[idx//2, idx%2].plot(mesh.vectorCCx, np.exp(x.x), "r-")
    ax_ls[idx//2, idx%2].legend(("True Model", "Recovered Model"))
    # ax[1].set_ylim([-2, 2])
    ax_ls[idx//2, idx%2].set_title("Wavelet-type " + reg.wavelets.wav)
plt.tight_layout()
plt.show()
##

