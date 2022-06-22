.. wbi documentation master file, created by
   sphinx-quickstart on Fri May  6 11:16:11 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to wabi's documentation!
===============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Installation
------------

WBI is available through `PyPI <https://pypi.org/project/wabi/>`_, and may be installed using ``pip``: ::

   $ pip install wbi

Contents
-----------------

.. toctree::
   :maxdepth: 2

   modules.rst
   examples.rst

wavelet-based-inversion
==================

Scale-dependent wavelet-based regularization scheme for geophysical 1D  inversion.

This flexible inversion scheme allows to easily obtain blocky, smooth and intermediate inversion models.
Different inversion models are obtained by simply changing the wavelet basis.
- db1: blocky inversion models
- db2-db4: sharper inversion models
- db5+: smoother inversion models

Daubechies (db) wavelets are ideal (see Deleersnyder et al, 2021), however, other wavelets can also be used. Simply run `pywt.wavelist()` to list the available options. The shape of the wavelet basis function (e.g., look `here <http://wavelets.pybytes.com>`_) is an indication of the type of minimum-structure the regularization method will promote.

- Fits within the modular SimPEG framework (see `SimPEG website <https://simpeg.xyz/>`_) (see examples)
- Fits within your own inversion codes (see examples with `empymod <https://empymod.emsig.xyz/en/stable/>`_)

How to cite
==================

Deleersnyder, W., Maveau, B., Hermans, T., & Dudal, D. (2021). Inversion of electromagnetic induction data using a novel wavelet-based and scale-dependent regularization term. Geophysical Journal International, 226(3), 1715-1729.  DOI: https://doi.org/10.1093/gji/ggab182

Open Access version on `ResearchGate <https://www.researchgate.net/publication/351407378_Inversion_of_electromagnetic_induction_data_using_a_novel_wavelet-based_and_scale-dependent_regularization_term>`_

The code
--------
Deleersnyder, W., Thibaut, R., 2022. WBI - Scale-dependent 1D wavelet-based inversion in Python
Questions?
Contact us on GitHub!
- `Wouter Deleersnyder <https://github.com/WouterDls>`_
- `Robin Thibaut <https://github.com/RobinThibaut>`_
