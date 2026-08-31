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

Before jumping into the code, check us out on [YouTube!](https://www.youtube.com/watch?v=b7NRddDScdM)

## Installation

Currently, SwamrRL is only available from source so it must be installed within the local
directory.

```sh
git clone https://github.com/SwarmRL/SwarmRL.git
cd SwarmRL
python -m pip install .
```

## Engines

SwarmRL drives a simulation through an `Engine`. Two are shipped:

* `swarmrl.engine.espresso` -- ESPResSo molecular dynamics, the reference backend.
* `swarmrl.engine.mujoco_engine` -- MuJoCo rigid-body dynamics, for scenes with
  contacts, walls, articulated agents or objects that have to be physically pushed.

### MuJoCo

MuJoCo is an optional dependency:

```sh
python -m pip install ".[mujoco]"
```

Bodies carrying a free joint are exposed to SwarmRL as `Colloid`s, and the `Action`s
returned by the `ForceFunction` are applied as Cartesian force and torque through
`xfrc_applied`, so the model needs no actuators:

```python
from swarmrl.engine.mujoco_engine import MujocoEngine, build_swarm_xml

engine = MujocoEngine(
    build_swarm_xml(n_agents=10),
    particle_types=0,      # must match the particle_type of your agents
    steps_per_slice=25,    # MuJoCo steps between two decisions
    planar=True,           # keep a 2D system 2D
)
engine.integrate(n_slices=100, force_model=force_function)
engine.render_trajectory("run.mp4")   # needs a GL backend
```

`build_swarm_xml` is a convenience arena of pucks in a box; pass your own MJCF string
or `.xml` path for anything else. Subclasses can hook into a slice through
`_pre_slice`, `_pre_step` and `_post_step` to advance environment state alongside the
physics.

Colloids in ESPResSo are thermal, so a bare MuJoCo model is not comparable to one.
`swarmrl.engine.mujoco_thermostat` supplies the missing fluctuating half of the
Langevin equation, at the amplitude the fluctuation-dissipation theorem demands, on
every damped degree of freedom. Because a MuJoCo model is usually written in
arbitrary units, the temperature is carried across by matching the dimensionless
Peclet number rather than by copying a value:

```python
from swarmrl.engine.mujoco_thermostat import LangevinThermostat, peclet_matched_kT

kT = peclet_matched_kT(drive_force=8.0, damping=3.0, radius=0.07, temperature=311.15)
engine = MujocoEngine(xml, thermostat=LangevinThermostat(kT=kT, seed=0))
```

`kT` is linear in the temperature and vanishes at `T = 0`, so a temperature sweep
behaves the way an ESPResSo one does.

Two things to know when rendering: set `MUJOCO_GL=egl` on a headless machine, and an
`EGLError` printed from `GLContext.__del__` at teardown is cosmetic -- the frames are
still written.

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
