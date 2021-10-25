"""
Configuration file for the package.
"""
import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="SwarmRL",
    version="0.0.1",
    author="Samuel Tovey and Christoph Lohrmann",
    author_email="tovey.samuel@gmail.com",
    description="A tool to study particle motion of reinforcement learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=required,
)
