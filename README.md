[![DOI](https://zenodo.org/badge/487773353.svg)](https://zenodo.org/badge/latestdoi/487773353)
[![Documentation Status](https://readthedocs.org/projects/1d-wavelet-based-inversion/badge/?version=latest)](https://1d-wavelet-based-inversion.readthedocs.io/en/latest/?badge=latest)


# wavelet-based-inversion
Scale-dependent wavelet-based regularization scheme for geophysical 1D  inversion

![Ensemble of inversion models](docs/img/Ensemble.png)

*This flexible inversion scheme allows to easily obtain blocky, smooth and intermediate inversion models. 
Different inversion models are obtained by simply changing the wavelet basis.*
- db1: blocky inversion models
- db2-db4: sharper inversion models
- db5+: smoother inversion models 

Daubechies (db) wavelets are ideal (see Deleersnyder et al, 2021), however, other wavelets can also be used. Simply run pywt.wavelist() to list the available options. The shape of the wavelet basis function (e.g., look [here](http://wavelets.pybytes.com/)) is an indication of the type of minimum-structure the regularization method will promote.
### Easy to use
- Fits within the modular SimPEG framework (see [SimPEG website](https://simpeg.xyz/)) (see examples)
- Fits within your own inversion codes (see examples with [empymod](https://empymod.emsig.xyz/en/stable/))

#### Documentation

https://1d-wavelet-based-inversion.readthedocs.io/en/latest/

### How to cite
**The method:**

Deleersnyder, W., Maveau, B., Hermans, T., & Dudal, D. (2021). Inversion of electromagnetic induction data using a novel wavelet-based and scale-dependent regularization term. _Geophysical Journal International_, 226(3), 1715-1729.  DOI: https://doi.org/10.1093/gji/ggab182

Open Access version on [ResearchGate](https://www.researchgate.net/publication/351407378_Inversion_of_electromagnetic_induction_data_using_a_novel_wavelet-based_and_scale-dependent_regularization_term)

**The code:**

Wouter Deleersnyder, & Robin Thibaut. (2022). WouterDls/1D-wavelet-based-inversion: Wavelet-Based Inversion (0.1.0). Zenodo. https://doi.org/10.5281/zenodo.6552695

### Questions?
Contact us on GitHub!
- [Wouter Deleersnyder](https://github.com/WouterDls)
- [Robin Thibaut](https://github.com/RobinThibaut)
