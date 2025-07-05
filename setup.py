from setuptools import setup
from setuptools_rust import RustExtension, Binding
import os



setup(
    name="source_modelling",
    packages=["source_modelling"],
    rust_extensions=[RustExtension("source_modelling.srf_parser", "source_modelling/srf_parser/Cargo.toml", binding=Binding.PyO3, native=True)],
    include_package_data=True,
    use_scm_version=True,
    zip_safe=False,  # required for Rust extensions
)
