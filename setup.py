"""
Configuration file for the package.
"""

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="swarmrl",
    version="0.0.1",
    author="Samuel Tovey and Christoph Lohrmann",
    author_email="tovey.samuel@gmail.com",
    description="A tool to study particle motion of reinforcement learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=required,
    # extras_require={
    #     "rnd": [
    #         "znnl @git+https://github.com/zincware/ZnNL.git",
    #     ]
    # },
)
