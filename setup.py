from setuptools import find_packages, setup
my_pckg = find_packages(include=["wbi"])
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()
setup(
    name="wabi",
    version="0.2.0",
    packages=my_pckg,
    include_package_data=True,
    url="https://github.com/WouterDls/1D-wavelet-based-inversion",
    license="BSD-3",
    author="Wouter Deleersnyder",
    author_email="wdls@eoas.ubc.ca",
    description="Wavelet-based regularization scheme 1D inversion",
    long_description=LONG_DESCRIPTION,
    install_requires=["numpy", "scipy", "cython", "simpeg==0.22.*", "discretize>=0.10", "PyWavelets"],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires=">=3.8",
)
