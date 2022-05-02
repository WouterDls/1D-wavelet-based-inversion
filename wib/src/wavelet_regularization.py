import warnings as wa

import numpy as np

# Install PyWavelets
import pywt

# Install SimPEG 0.14.0 or higher
# This all comes with SimPEG > 0.14.0
from SimPEG import utils
from SimPEG.regularization import BaseRegularization
from scipy.sparse import eye


class WaveletRegularization1D(BaseRegularization):
    """
    Base Simple Smooth Regularization. This base class regularizes on the first
    spatial derivative, not considering length scales, in the provided
    orientation

    **Optional Inputs**

    :param discretize.base.BaseMesh mesh: SimPEG mesh
    :param int nP: number of parameters
    :param IdentityMap mapping: regularization mapping, takes the model from model space to the space you want to regularize in
    :param numpy.ndarray mref: reference model
    :param numpy.ndarray indActive: active cell indices for reducing the size of differential operators in the definition of a regularization mesh
    :param numpy.ndarray cell_weights: cell weights
    :param bool mrefInSmooth: include the reference model in the smoothness computation? (eg. look at Deriv of m (False) or Deriv of (m-mref) (True))
    :param numpy.ndarray cell_weights: vector of cell weights (applied in all terms)
    """

    def __init__(self, mesh, orientation="x", wav="db1", **kwargs):
        """
        Parameters
        ----------
        mesh
        orientation
        wav
        kwargs
        """
        self.orientation = orientation
        self.p = 1  # See Deleersnyder et al., 2021 for details.
        self.eps = 1e-8  # perturbing parameter, default is 1e-8. Should be smaller than 1e-4.
        self.mesh = mesh
        self.mrefInSmooth = False
        assert self.orientation in [
            "x",
            "y",
            "z",
        ], "Orientation must be 'x', 'y' or 'z'"

        if self.orientation == "x":
            self.wavelets = Wavelet(wav, mesh.nCx, **kwargs)

        elif self.orientation == "y":
            self.wavelets = Wavelet(wav, mesh.nCy, **kwargs)
            assert mesh.dim > 1, (
                "Mesh must have at least 2 dimensions to regularize along the "
                "y-direction"
            )

        elif self.orientation == "z":
            self.wavelets = Wavelet(wav, mesh.nCz, **kwargs)
            assert mesh.dim > 2, (
                "Mesh must have at least 3 dimensions to regularize along the "
                "z-direction"
            )
        # Generate the the scale-dependent-weights for each coefficient in X
        self.R = self._regularization_matrix()
        super(WaveletRegularization1D, self).__init__(mesh=mesh, **kwargs)

    @property
    def _multiplier_pair(self):
        return "alpha_{orientation}".format(orientation=self.orientation)

    @utils.timeIt
    def __call__(self, m):
        """
        We use a $\ell_1$ perturbed Ekblom measure as differentiable sparsity measure.

        .. math::

            r(m) =  \sqrt{m^TW^TWm + \epsilon}

        """

        # Do the wavelet transform on each 1D vector in the tensor.
        X = self.wavelets.W @ m.reshape(-1, 1)
        return np.sum(self.R * np.sqrt(X ** 2 + self.eps))  # the actual measure

    @utils.timeIt
    def deriv(self, m):
        """

        The regularization in wavelet domain is:

        .. math::

            R(x) =  \sum_j \sqrt{x_j^2 + \epsilon}

        So the derivative is straight forward:

        .. math::

            \frac{\partial R(x)}{\partial x_j} =  \sum_j \frac{x_j}{\sqrt{x_j^2 + \epsilon}}

        And using the chain rule to model space:

        .. math::

            \frac{\partial R(m)}{\partial m_j} = \frac{\partial R(m)}{\partial x_i}\frac{\partial x_i}{\partial m_j}
            with \frac{\partial x}{\partial m} = W

        """
        mD = self.mapping.deriv(m)
        # Do the wavelet transform
        X = self.wavelets.W @ m.reshape(-1, 1)
        # Generate derivative w.r.t. x
        deriv_x = self.R * X / np.sqrt(X ** 2 + self.eps)
        # Chain rule w.r.t. m
        deriv_m = self.wavelets.W.T @ deriv_x
        return (mD.T * deriv_m).flatten()  # Chain rule w.r.t. SimPEG mapping

    @utils.timeIt
    def deriv2(self, m, v=None):
        """
        Second derivative

        The second derivative of the perturbed Ekblom measure is highly unstable for small epsilon. Most methods do not
        use Hessian information, except for preconditioning (see e.g., optimization -> InexactGaussNewton) . Therefore,
        the unit matrix is used as Hessian. This results in a more stable optimization routine.

        :param numpy.ndarray m: geophysical model
        :param numpy.ndarray v: vector to multiply
        :rtype: scipy.sparse.csr_matrix
        :return: WtW, or if v is supplied WtW*v (numpy.ndarray)

        """
        mD = self.mapping.deriv(m)
        if v is None:
            return mD.T * eye(m.size)
        return mD.T * v

    def _generate_scale_dependency_vector(self, wavelet):
        """
        Wavelet-coefficients corresponding to small-scale effects of the model are penalized more heavily.
        The scaling coefficient is never zero, so no regularization on the scaling coefficients.
        .. math::
            x = [v_{0,0}, w_{0,0}, w_{1,0},w_{1,1}, w_{2,1},w_{2,1}, \cdots, w_{n,k}, \cdots ]

            \phi_m(x) =  \frac{1}{E} \sum_{n}^{N} 2^n\sum_k \mu(w_n,k)
        :param wavelet:contains info about the specific wavelet-transform
        :return:
        """
        # Do wavelet decomposition (=transform)
        coeffs = pywt.wavedec(np.ones(wavelet.n_m), wavelet.wav, level=wavelet.DWTlevel)
        # returns a list of lists with scaling/wavelet coefficients per scale of resolution
        scale_dependency_vector = np.hstack([2**(j*self.p)*np.ones(c.shape) for j, c in enumerate(coeffs)])
        scale_dependency_vector[:coeffs[0].size] = 0 # No regularization on scaling coefficients
        return scale_dependency_vector.reshape(-1, 1)/np.linalg.norm(scale_dependency_vector) # Normalization, only valid vor 1D inversion (as in Deleersnyder et al, 2021)

    def _regularization_matrix(self):
        """

        :return:
        """
        if self.mesh.dim == 1:
            n = 1
        else:
            raise NotImplementedError('Future release')
        scale_dependency_vector = self._generate_scale_dependency_vector(self.wavelets)
        return np.tile(scale_dependency_vector.reshape(-1, 1), (1, n))


class Wavelet:
    """
    The object containing all specific functionalities for a wavelet type.

    Parameters
    ----------
    wav : string
          The wavelet type (typically family + str(number of vanishing moments) e.g. Daubechies 1 --> db1)
    n_m : integer
          The number of model parameters in model space (i.e. the length of vector m)
    DWTlevel : integer
               The level of the discrete wavelet transform
    signal_extension : string
                      The type/mode of signal extension, used in the discrete wavelet transform


    Attributes
    ----------
    W   : numpy array of size n_x by n_m
          Transformation matrix of wavelet transform

    See Also
    --------
    ...
    """

    def __init__(self, wav, n_m, DWTlevel=None, signal_extension=None):
        self.W = None

        self.n_m = n_m
        self.wav = wav
        self.DWTlevel = DWTlevel
        self.signal_extension = signal_extension

        self._update_W()
        if self.W is None:
            raise Exception('Init failed')
        self.n_x = self.W.shape[0]

    @property
    def n_m(self):
        """ The number of model parameters (in model space) """
        return self._n_m

    @n_m.setter
    def n_m(self, n):
        if n <= 0:
            raise Exception('n_m are the number of model parameters, thus strictly positive')
        else:
            self._n_m = n
        if self.W is not None:
            self._update_W()

    @property
    def wav(self):
        """ Wavelet family to use. See Deleersnyder et al, 2021 for the rationale behind the choice of the optimal wavelet
        In general, Daubechies (db) wavelets are prefered.
        db1 yields blocky inversion models
        db2-db4 yield inversion models with sharp interfaces
        db5+ yield smooth inversion models
        The discretization of the inversion model also plays a role.
        Changing the discretization may affect the optimal 'choice' for the wavelet.
        """
        if self._wav is None:
            raise Exception('The wavelet basis function is None')
        else:
            return self._wav

    @wav.setter
    def wav(self, type):
        """ """
        if type not in pywt.wavelist():
            raise Exception("unknown wavelet type, use names from " + str(pywt.wavelist()))
        else:
            self._wav = type
        if self.W is not None:
            self._update_W()

    @property
    def DWTlevel(self):
        """ The level of decomposition of the discrete wavelet transform"""
        if self._DWTlevel is None:
            raise Exception('The discrete wavelet transform level (DWTlevel) is None')
        else:
            return self._DWTlevel

    @DWTlevel.setter
    def DWTlevel(self, level):
        maxlevel = pywt.dwt_max_level(self.n_m, self.wav)
        if level is None:
            self._DWTlevel = maxlevel
        else:
            if level > maxlevel:
                wa.warn(
                    "Boundary effects: The user-defined DWTlevel exceeds the suggested maximum DWT level of "
                    + str(maxlevel)
                )
            self._DWTlevel = level
        if self.W is not None:
            self._update_W()

    @property
    def signal_extension(self):
        """ Due to the cascading filter banks algorithm, an extrapolation method is required. Choose the method which
        introduces the least artifacts. In general, "smooth" is a good choice. """
        if self._signal_extension is None:
            raise Exception('The signal extension type is None')
        else:
            return self._signal_extension

    @signal_extension.setter
    def signal_extension(self, type):
        if type is None:
            self._signal_extension = "smooth"
        elif type not in pywt.Modes.modes:
            raise Exception("Typo in signal extension, choose from " + pywt.Modes.modes)
        else:
            self._signal_extension = type
        if self.W is not None:
            self._update_W()

    def _update_W(self):
        self.W = np.hstack(
            pywt.wavedec(np.eye(self.n_m), self.wav, level=self.DWTlevel, mode=self.signal_extension, axis=1)
        ).T