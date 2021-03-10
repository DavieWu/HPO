from setuptools import setup, find_packages

setup(
    name="frankensteins-automl",
    version="0.0.1",
    description="AutoML tool with stitched together optmizers.",
    author="Lukas Brandt",
    url="https://github.com/Berberer/frankensteins-automl",
    packages=find_packages(exclude=("tests", "docs")),
)
