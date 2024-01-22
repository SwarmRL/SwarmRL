"""
Configuration file for the package.
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="swarmrl",
    version="0.0.1",
    author="Samuel Tovey and Christoph Lohrmann",
    author_email="tovey.samuel@gmail.com",
    description="A tool to study particle motion of reinforcement learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=required,
    extras_require={"rnd": ["znnl @ git+https://github.com/zincware/ZnNL.git"]},
)
