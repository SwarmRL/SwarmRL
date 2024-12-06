![PyTest](https://github.com/SwarmRL/SwarmRL/actions/workflows/pytest.yml/badge.svg)
[![codecov](https://codecov.io/gh/SwarmRL/SwarmRL/branch/master/graph/badge.svg)](https://codecov.io/gh/SwarmRL/SwarmRL)
[![code-style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black/)

# SwarmRL
SwarmRL is a Python package aimed at providing a simple and flexible framework for
developing and testing reinforcement learning and mathematical optimization algorithms
for multi-agent systems.
Some key features of SwarmRL include:

* Integration with EspressoMD simulation engine
* Support for experimental setups
* Actor-critic reinforcement learning strategies

## Installation

Currently, SwamrRL is only available from source so it must be installed within the local
directory.

```sh
git clone https://github.com/SwarmRL/SwarmRL.git
cd SwarmRL
python -m pip install .
```

## Looking for a Starting Point?

Our documentation is a work in progress but can be found [here](swarmrl.github.io/SwarmRL.ai/).
If you have questions about the code or find any problems, please create an issue so we can work on it as soon as possible.
If you're feeling adventurous, you can check out our custom-built Swarm GPT, [here](https://chat.openai.com/g/g-3lniVEMpK-swarm-gpt) which has been conditioned on the SwarmRL repository and will be updated as more resources become available. Be careful though! It isn't perfect but not a bad place to start for general principles of reinforcement learning and pieces of the software.

## Contributing

Install developer dependencies:

```sh
python -m pip install -r dev-requirements.txt
python -m pip install ".[rnd]"
python -m pip install sphinx sphinx_rtd_theme
```

Run the linters and code formatters with pre-commit:

```sh
pre-commit run --all-files
```

Build the documentation with sphinx:

```sh
cd docs/
make html
xdg-open build/html/index.html
```

Run the testsuite with pytest and CTest:

```sh
# run SwarmRL testsuite
pytest --ignore CI/espresso_tests
# run ESPResSo testsuite
sh CI/run_espresso_test_suite.sh -j $(nproc)
```

The ESPResSo testsuite leverages CTest to schedule jobs in parallel.
The wrapper script assumes the ESPResSo package is part of the `$PYTHONPATH`
environment variable or available in the current Python virtual environment.
Additional CTest flags can be passed to the wrapper script,
such as `-LE long` to skip integration tests.

When contributing new features, consider adding a unit test in the `CI/unit_tests/` folder.
These tests are automatically discovered by the pytest test driver.
For ESPResSo tests, the CTest test driver is used instead;
add the test in one of the `CI/espresso_tests/` subfolders and
add a corresponding line in the `CI/espresso_tests/CTestTestfile.cmake` file.

To run code coverage locally:

```sh
COVERAGE=1 sh CI/run_espresso_test_suite.sh -j $(nproc)
python -m coverage run --parallel-mode -m pytest --ignore CI/espresso_tests
python -m coverage combine . CI/espresso_tests
python -m coverage html --omit="*/espressomd/*" --directory=coverage_html
xdg-open coverage_html/index.html
```
