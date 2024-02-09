"""
Configuration file for the package.
"""

from setuptools import setup, find_packages

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
    python_requires=">=3.9",
    install_requires=required,
    extras_require={"rnd": 
                    ["znnl @ git+https://github.com/zincware/ZnNL.git@Konsti_fix_requirements#egg=znnl-0.0.1",]
                    },
)
