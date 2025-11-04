# ===== ./setup.py =====
from setuptools import setup, find_packages

setup(
    name="tabkan",
    version="1.0.1",
    author="Ali Eslamian, Alireza Afzal Aghaei, Qiang Cheng",
    author_email="ali.eslamian@uky.edu",
    description="TabKAN: A Framework for Advancing Tabular Data Analysis using Kolmogorov-Arnold Networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aseslamian/TabKAN",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.0",
        "numpy>=1.22",
        "scikit-learn",
        "optuna>=3.0",
        "pandas",
        "tqdm",
        "pykan",  # For the baseline Spline KAN
        "fkan",   # For FractionalKAN
        "rkan"    # For JacobiRKAN and PadeRKAN
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
)
