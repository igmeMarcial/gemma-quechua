from setuptools import find_packages, setup

setup(
    name="gemma-quechua",
    version="0.1.0",
    description="Pretraining pipeline for Quechua using Unsloth and Gemma",
    author="Marcial Igme",
    author_email="igmemarcial@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "PyMuPDF",
        "datasets",
        "nltk",
        "unsloth",
    ],
)
