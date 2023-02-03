import os

import numpy as np
from SimPEG import (
    simulation,
    maps,
    data_misfit,
    directives,
    optimization,
    inverse_problem,
    inversion,
)
from discretize import TensorMesh

from wbi import wavelet_regularization as regularization

my_path = os.path.dirname(os.path.abspath(__file__))


def test_wavelet_regularization_1d():
    """
    Test the wavelet regularization in 1D
    """

    nParam = 100

    mesh = TensorMesh([nParam])

    true_model = np.zeros(mesh.nC)
    true_model[mesh.vectorCCx > 0.3] = 1.0
    true_model[mesh.vectorCCx > 0.45] = -0.5
    true_model[mesh.vectorCCx > 0.6] = 0

    model_map = maps.IdentityMap(mesh)

    nData = 20

    jk = np.linspace(1.0, 60.0, nData)
    p = -0.25
    q = 0.25

    def g(k):
        return np.exp(p * jk[k] * mesh.vectorCCx) * np.cos(
            np.pi * q * jk[k] * mesh.vectorCCx
        )

    G = np.empty((nData, nParam))
    for i in range(nData):
        G[i, :] = g(i)

    sim = simulation.LinearSimulation(mesh, G=G, model_map=model_map)
    std = 0.01
    np.random.seed(1)

    data_obj = sim.make_synthetic_data(
        true_model, relative_error=std, add_noise=True
    )  # noqa

    dmis = data_misfit.L2DataMisfit(simulation=sim, data=data_obj)

    reg = regularization.WaveletRegularization1D(mesh, wav="db3")
    opt = optimization.InexactGaussNewton(maxIter=100, maxIterLS=20)

    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    target_misfit = directives.TargetMisfit()

    directives_list = [target_misfit]

    inv_prob.beta = 1e4
    inv = inversion.BaseInversion(inv_prob, directives_list)

    np.random.seed(1)
    starting_model = np.random.rand(nParam) * 0.1

    # Run inversion
    recovered_model = inv.run(starting_model)

    # load the reference model
    ref_model = np.load(os.path.join(my_path, "data", "ref_model_1d.npy"))

    # Check that the recovered model is close to the reference model
    assert np.allclose(recovered_model, ref_model, atol=0.1)


if __name__ == "__main__":
    test_wavelet_regularization_1d()