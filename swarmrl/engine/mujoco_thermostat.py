"""
Langevin thermostat for the MuJoCo engine.

This is ESPResSo's ``system.thermostat.set_langevin`` transplanted into MuJoCo. The
two engines already share the dissipative half of the Langevin equation -- ESPResSo
applies ``-gamma * v`` per particle, MuJoCo applies ``-dof_damping[i] * qvel[i]`` per
degree of freedom -- so the only missing piece is the fluctuating half at the
amplitude the fluctuation-dissipation theorem demands:

    m dv/dt = F_ext - gamma v + sqrt(2 gamma kT / dt) xi,     xi ~ N(0, 1)

ESPResSo draws its kick from a uniform distribution, ``sqrt(24 gamma kT / dt) *
(U(0,1) - 0.5)``, which has exactly the same variance ``2 gamma kT / dt``. Over the
many steps that set any measurable quantity the two are interchangeable by the
central limit theorem, so a Gaussian draw is used here.

Working in GENERALIZED coordinates (``data.qfrc_applied``, indexed by
``model.dof_damping``) rather than Cartesian ones is what keeps this uniform across
joint types: free-jointed swimmers and hinge-mounted agents are handled by the same
line, and a hinge's angular DoF gets the right units for free because
``dof_damping`` is already expressed per generalized velocity.

Two conventions are inherited from ESPResSo deliberately:

* A DoF with zero damping gets no noise. ESPResSo pins a particle by giving it
  ``gamma=0``; the same trick works here, and it is also what FDT requires -- no
  dissipation, no fluctuation.
* The noise stream is seeded once and then runs continuously. It is NOT reseeded on
  ``reset()``, so successive episodes from one engine see independent noise rather
  than a replay of the same realisation, while a whole run stays reproducible from
  its seed.

Units
-----
A MuJoCo model is normally written in arbitrary units while ESPResSo runs in SI, so
copying a kT across would be meaningless. ``peclet_matched_kT`` maps between the two
via the dimensionless Peclet number instead, which is the one number that survives
the change of unit system.
"""

import typing

import numpy as np
from loguru import logger

try:
    import mujoco
except ModuleNotFoundError:
    logger.warning("Could not find mujoco. Features will not be available")

# Boltzmann constant, J/K. Only used to reduce the ESPResSo side to a Peclet number.
BOLTZMANN = 1.380649e-23

# One worked example of an ESPResSo swimmer, used as the default reference point of
# `espresso_peclet`: the microrobots of Heuthe et al., 3 um radius, swimming at
# 0.6 um/s in water at body temperature. These are defaults for convenience, not a
# statement about what a SwarmRL simulation must look like -- pass your own.
REFERENCE_TEMPERATURE = 311.15  # K
REFERENCE_RADIUS = 3.0e-6  # m
REFERENCE_SWIM_VELOCITY = 0.6e-6  # m/s
REFERENCE_DYN_VISCOSITY = 1e-3  # Pa s, water


def stokes_gammas(dyn_viscosity: float, radius: float) -> typing.Tuple[float, float]:
    """
    Stokes drag coefficients of a sphere, as SwarmRL computes them.

    Deliberately a standalone copy of ``_calc_friction_coefficients`` in
    ``swarmrl/engine/espresso.py``: that one is a private method on ``EspressoMD`` and
    reaching it would make the MuJoCo engine import ESPResSo, which is the dependency
    this module exists to avoid. It is two textbook one-liners, so the duplication is
    cheaper than the coupling.

    Parameters
    ----------
    dyn_viscosity : float
            Dynamic viscosity of the fluid, Pa s.
    radius : float
            Sphere radius, m.

    Returns
    -------
    gamma_translation, gamma_rotation : float
            Translational (kg/s) and rotational (kg m^2/s) friction coefficients.
    """
    return (
        6 * np.pi * dyn_viscosity * radius,
        8 * np.pi * dyn_viscosity * radius**3,
    )


def espresso_peclet(
    temperature: float = REFERENCE_TEMPERATURE,
    radius: float = REFERENCE_RADIUS,
    swim_velocity: float = REFERENCE_SWIM_VELOCITY,
    dyn_viscosity: float = REFERENCE_DYN_VISCOSITY,
) -> float:
    """
    Translational Peclet number of an ESPResSo swimmer, ``Pe = v a / D``.

    ``D = kT / (6 pi eta a)``. This is the one number that survives the change of
    unit system: it is the ratio of active to diffusive transport, and matching it
    is what makes MuJoCo agents feel the same amount of noise per unit of directed
    motion as the ESPResSo ones do.

    Parameters
    ----------
    temperature : float
            Thermostat temperature, K.
    radius : float
            Swimmer radius, m.
    swim_velocity : float
            Swim speed, m/s.
    dyn_viscosity : float
            Dynamic viscosity, Pa s.

    Returns
    -------
    peclet : float
            The dimensionless Peclet number.
    """
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    gamma, _ = stokes_gammas(dyn_viscosity, radius)
    diffusion = BOLTZMANN * temperature / gamma
    if diffusion == 0.0:
        # T = 0 is a real point in a temperature sweep, not a degenerate one: no
        # diffusion at all, so transport is infinitely Peclet-dominated.
        # peclet_matched_kT maps this back to kT = 0.
        return np.inf
    return swim_velocity * radius / diffusion


def peclet_matched_kT(
    drive_force: float,
    damping: float,
    radius: float,
    peclet: float = None,
    **espresso_kwargs,
) -> float:
    """
    kT in MuJoCo units that reproduces an ESPResSo Peclet number.

    A MuJoCo agent's steady swim speed is ``v = drive_force / damping`` (its joint is
    damped, so a constant applied force gives a terminal velocity exactly as Stokes
    drag does), and its diffusion coefficient under this thermostat is
    ``D = kT / damping``. Setting ``v a / D = Pe`` gives

        kT = drive_force * radius / Pe

    Parameters
    ----------
    drive_force : float
            Magnitude of the propulsion force, MuJoCo force units. This is the
            ``force`` of the agent's forward ``Action``.
    damping : float
            Translational damping of the agent's joint. Cancels out of the result but
            is kept in the signature because it defines the speed the match is made
            at, and a caller changing one without the other is almost certainly
            making a mistake.
    radius : float
            Agent radius, MuJoCo length units.
    peclet : float
            Target Peclet number. Defaults to ``espresso_peclet(**espresso_kwargs)``.
    **espresso_kwargs
            Forwarded to ``espresso_peclet`` when ``peclet`` is None -- notably
            ``temperature`` for a temperature sweep.

    Returns
    -------
    kT : float
            Thermostat temperature in MuJoCo energy units.
    """
    if peclet is None:
        peclet = espresso_peclet(**espresso_kwargs)
    if peclet <= 0.0:
        raise ValueError("Peclet number must be positive; use kT=0 for no noise.")
    if not np.isfinite(peclet):
        return 0.0  # the T = 0 limit
    velocity = drive_force / damping
    return velocity * radius * damping / peclet


class LangevinThermostat:
    """
    Per-DoF Langevin noise applied through ``data.qfrc_applied``.

    The kick is redrawn every MuJoCo step, not every RL slice. That matters whenever
    the system is not deeply overdamped: if the momentum relaxation time ``m/gamma``
    is longer than a slice, per-slice noise would be visibly coloured rather than
    white.

    ``qfrc_applied`` is a different accumulator from the ``xfrc_applied`` the policy
    actions are written to, so the thermostat never has to save and restore the
    action force. It clears only the DoFs it owns, leaving any other generalized
    force a caller applies untouched.

    Attributes
    ----------
    kT : float
            Temperature in MuJoCo energy units. Zero makes ``apply`` a no-op.
    dofs : np.ndarray
            Indices of the thermostatted degrees of freedom, set by ``bind``.
    sigma : np.ndarray
            Per-DoF noise amplitude ``sqrt(2 c kT / dt)``, set by ``bind``.
    """

    def __init__(
        self,
        kT: float,
        seed: int = 0,
        planar: bool = True,
        thermalise_rotation: bool = True,
    ):
        """
        Parameters
        ----------
        kT : float
                Temperature in MuJoCo energy units (mass * length^2 / time^2). Use
                ``peclet_matched_kT`` to derive it from an ESPResSo temperature.
        seed : int
                Seed of the thermostat's own RNG. Kept separate from the environment
                RNG on purpose: a temperature sweep can then hold the scene layout
                and the spawn positions fixed while only the noise changes.
        planar : bool
                If True, only the in-plane DoFs of a free joint are thermostatted
                (x, y and the yaw). Kicking z or the tilt DoFs would make a body
                resting on a floor hop against the contact instead of diffusing
                across it.
        thermalise_rotation : bool
                If False, the rotational DoFs are left out entirely and only
                translation is thermostatted. Useful when a model's rotational
                damping does not follow the Stokes ratio ``(4/3) a^2`` and the
                rotational match to ESPResSo would therefore be wrong anyway.
        """
        self.kT = float(kT)
        self.planar = planar
        self.thermalise_rotation = thermalise_rotation
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        self.dofs = np.zeros(0, dtype=int)
        self.sigma = np.zeros(0)

    def bind(self, model: "mujoco.MjModel", body_ids: np.ndarray, timestep: float):
        """
        Work out which DoFs to thermostat and precompute their noise amplitudes.

        Parameters
        ----------
        model : mujoco.MjModel
                Compiled model; supplies ``dof_damping`` and the joint layout.
        body_ids : np.ndarray
                Bodies to thermostat -- normally the engine's agent bodies.
        timestep : float
                MuJoCo integration timestep. The noise amplitude carries
                ``1/sqrt(dt)`` because the kick approximates a delta-correlated
                force.
        """
        dofs = []
        for bid in np.atleast_1d(body_ids):
            for k in range(model.body_jntnum[bid]):
                jid = model.body_jntadr[bid] + k
                adr = model.jnt_dofadr[jid]
                jtype = model.jnt_type[jid]

                if jtype == mujoco.mjtJoint.mjJNT_FREE:
                    local = [0, 1] if self.planar else [0, 1, 2]
                    if self.thermalise_rotation:
                        local += [5] if self.planar else [3, 4, 5]
                elif jtype == mujoco.mjtJoint.mjJNT_BALL:
                    local = [0, 1, 2] if self.thermalise_rotation else []
                else:  # hinge or slide: a single DoF
                    local = [0]
                dofs.extend(adr + i for i in local)

        dofs = np.array(sorted(set(dofs)), dtype=int)
        damping = model.dof_damping[dofs]

        # ESPResSo's convention: gamma = 0 excludes a particle from the thermostat.
        # It is also what FDT requires -- an undamped DoF has no dissipation to
        # balance, so noise on it would heat without bound.
        keep = damping > 0.0
        self.dofs = dofs[keep]
        self.sigma = np.sqrt(2.0 * damping[keep] * self.kT / timestep)

    def apply(self, data: "mujoco.MjData"):
        """
        Draw a fresh kick and write it into ``data.qfrc_applied``.

        Call once per MuJoCo step, before ``mj_step``.
        """
        if self.kT == 0.0 or self.dofs.size == 0:
            return
        # Overwrite, rather than accumulate, the previous step's kick -- but only on
        # the DoFs this thermostat owns.
        data.qfrc_applied[self.dofs] = self.sigma * self.rng.standard_normal(
            self.dofs.size
        )

    def expected_variance(
        self, model: "mujoco.MjModel", data: "mujoco.MjData"
    ) -> np.ndarray:
        """
        Stationary velocity variance this thermostat actually produces, per DoF.

        MuJoCo's ``implicitfast`` integrator treats joint damping implicitly, which
        biases the discrete-time result away from the continuum ``kT/m``. Propagating
        ``v <- (v + f dt/m) / (1 + c dt/m)`` to its fixed point gives

            <v^2> = (kT / m) / (1 + c dt / (2 m))

        The bias is small whenever ``c dt / m << 1``, but this is the number a
        calibration test should compare against rather than the idealised ``kT/m``.

        The inertia is read off the diagonal of the mass matrix. Note that
        ``model.dof_invweight0`` is NOT this quantity: it is an articulation-averaged
        inverse inertia and the two can disagree by ~10% on a rotational DoF.

        Parameters
        ----------
        model : mujoco.MjModel
                Compiled model.
        data : mujoco.MjData
                State whose mass matrix to read. Must have had ``mj_forward`` (or a
                step) run on it.

        Returns
        -------
        variance : np.ndarray
                Expected ``<v_i^2>`` for each DoF in ``self.dofs``, same order.
        """
        full = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, full)
        inertia = np.diag(full)[self.dofs]
        damping = model.dof_damping[self.dofs]
        timestep = model.opt.timestep
        return (self.kT / inertia) / (1.0 + 0.5 * damping * timestep / inertia)
