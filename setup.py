import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "source_modelling.srf_reader",
        ["source_modelling/srf_reader.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="source_modelling",
    ext_modules=cythonize(extensions),
)
