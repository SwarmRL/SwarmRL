"""
Calibration tests for the MuJoCo Langevin thermostat.

A thermostat that is merely "on" is worthless: what matters is whether it puts the
right amount of energy in, and whether the diffusion that comes out is the
``D = kT/gamma`` the ESPResSo side is built on. These tests measure both against
theory rather than asserting that noise exists.

MuJoCo is an optional dependency, so the whole module skips when it is absent.
"""

import unittest as ut

import numpy as np

try:
    import mujoco

    from swarmrl.engine.mujoco_thermostat import (
        LangevinThermostat,
        espresso_peclet,
        peclet_matched_kT,
        stokes_gammas,
    )

    MUJOCO_AVAILABLE = True
except ModuleNotFoundError:
    MUJOCO_AVAILABLE = False

# A representative operating point: an agent driven by a force of 8.0 against a joint
# damping of 3.0, with a radius of 0.07 model units, matched to the reference
# ESPResSo swimmer. Roughly kT = 0.024.
DRIVE, DAMPING, RADIUS = 8.0, 3.0, 0.07

# Independent walkers in one model: contype/conaffinity are cleared so they pass
# through each other, which turns one mj_step into N independent Langevin samples.
# A single walker's MSD slope scatters by several percent even over a long run, so
# the ensemble is what makes the diffusion test discriminating rather than loose.
N_WALKERS = 48


def free_particle_xml(damping: float, n: int = N_WALKERS, mass: float = 1.0) -> str:
    """
    MJCF with n non-interacting free-jointed particles and no gravity.

    Gravity and contacts are left out on purpose: floor friction is dissipation with
    no matching fluctuation term, and it would bias every measurement here.
    """
    bodies = "".join(
        f"""
    <body name="p{i}" pos="0 0 0">
      <joint name="p{i}_joint" type="free" damping="{damping}"/>
      <geom name="p{i}_geom" type="cylinder" size="0.07 0.07" mass="{mass}"
            contype="0" conaffinity="0"/>
    </body>"""
        for i in range(n)
    )
    return f"""
<mujoco model="free_particles">
  <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
  <worldbody>{bodies}
  </worldbody>
</mujoco>
"""


@ut.skipUnless(MUJOCO_AVAILABLE, "mujoco is not installed")
class TestPecletMatching(ut.TestCase):
    """
    The unit bridge between an ESPResSo temperature and a MuJoCo kT.
    """

    def test_stokes_gammas(self):
        """Textbook drag coefficients of a sphere."""
        gamma_t, gamma_r = stokes_gammas(1e-3, 3e-6)
        self.assertAlmostEqual(gamma_t, 6 * np.pi * 1e-3 * 3e-6)
        self.assertAlmostEqual(gamma_r, 8 * np.pi * 1e-3 * 3e-6**3)

    def test_peclet_round_trip(self):
        """kT from a Peclet number must reproduce that Peclet number."""
        target = espresso_peclet()
        kT = peclet_matched_kT(DRIVE, DAMPING, RADIUS, peclet=target)
        measured = (DRIVE / DAMPING) * RADIUS / (kT / DAMPING)
        self.assertAlmostEqual(measured / target, 1.0, places=9)

    def test_kT_is_linear_in_temperature(self):
        """A temperature sweep must behave like one."""
        warm = peclet_matched_kT(DRIVE, DAMPING, RADIUS, temperature=300.0)
        half = peclet_matched_kT(DRIVE, DAMPING, RADIUS, temperature=150.0)
        self.assertAlmostEqual(half / warm, 0.5, places=9)

    def test_zero_kelvin_is_athermal(self):
        """T = 0 is a real point in a sweep, not a division by zero."""
        self.assertTrue(np.isinf(espresso_peclet(temperature=0.0)))
        self.assertEqual(
            peclet_matched_kT(DRIVE, DAMPING, RADIUS, temperature=0.0), 0.0
        )

    def test_negative_temperature_raises(self):
        with self.assertRaises(ValueError):
            espresso_peclet(temperature=-1.0)

    def test_non_positive_peclet_raises(self):
        with self.assertRaises(ValueError):
            peclet_matched_kT(DRIVE, DAMPING, RADIUS, peclet=0.0)


@ut.skipUnless(MUJOCO_AVAILABLE, "mujoco is not installed")
class TestDofSelection(ut.TestCase):
    """
    Which degrees of freedom get kicked, and which are deliberately left alone.
    """

    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(free_particle_xml(DAMPING, n=2))
        self.data = mujoco.MjData(self.model)
        self.bodies = np.arange(1, 3)

    def bind(self, **kwargs) -> "LangevinThermostat":
        thermostat = LangevinThermostat(kT=0.05, seed=0, **kwargs)
        thermostat.bind(self.model, self.bodies, self.model.opt.timestep)
        return thermostat

    def test_planar_mode_picks_x_y_and_yaw(self):
        """Kicking z or the tilt DoFs would make a body hop against the floor."""
        thermostat = self.bind(planar=True)
        offsets = thermostat.dofs % 6
        np.testing.assert_array_equal(np.unique(offsets), [0, 1, 5])
        self.assertEqual(thermostat.dofs.size, 6)

    def test_three_dimensional_mode_picks_every_dof(self):
        thermostat = self.bind(planar=False)
        self.assertEqual(thermostat.dofs.size, 12)

    def test_rotation_can_be_left_out(self):
        """Useful when a model's rotational damping is not the Stokes ratio."""
        thermostat = self.bind(planar=True, thermalise_rotation=False)
        np.testing.assert_array_equal(np.unique(thermostat.dofs % 6), [0, 1])

    def test_undamped_dofs_are_excluded(self):
        """ESPResSo's gamma = 0 pinning convention; also what FDT demands."""
        model = mujoco.MjModel.from_xml_string(free_particle_xml(0.0, n=2))
        thermostat = LangevinThermostat(kT=0.05)
        thermostat.bind(model, np.arange(1, 3), model.opt.timestep)
        self.assertEqual(thermostat.dofs.size, 0)

    def test_hinge_joints_are_thermostatted(self):
        """The generalized-coordinate formulation must cover non-free joints."""
        xml = """
        <mujoco>
          <option timestep="0.002" gravity="0 0 0"/>
          <worldbody>
            <body name="arm" pos="0 0 0">
              <joint name="hinge" type="hinge" axis="0 0 1" damping="2.0"/>
              <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        thermostat = LangevinThermostat(kT=0.05)
        thermostat.bind(model, np.array([1]), model.opt.timestep)
        np.testing.assert_array_equal(thermostat.dofs, [0])

    def test_sigma_follows_the_fluctuation_dissipation_theorem(self):
        """sigma = sqrt(2 c kT / dt), per DoF."""
        thermostat = self.bind(planar=True)
        expected = np.sqrt(
            2.0
            * self.model.dof_damping[thermostat.dofs]
            * thermostat.kT
            / self.model.opt.timestep
        )
        np.testing.assert_allclose(thermostat.sigma, expected)


@ut.skipUnless(MUJOCO_AVAILABLE, "mujoco is not installed")
class TestKickApplication(ut.TestCase):
    """
    How the kick reaches MuJoCo.
    """

    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(free_particle_xml(DAMPING, n=2))
        self.data = mujoco.MjData(self.model)

    def thermostat(self, kT: float, seed: int = 0) -> "LangevinThermostat":
        thermostat = LangevinThermostat(kT=kT, seed=seed)
        thermostat.bind(self.model, np.arange(1, 3), self.model.opt.timestep)
        return thermostat

    def test_zero_kT_is_a_no_op(self):
        """The athermal default must not touch the simulation at all."""
        thermostat = self.thermostat(kT=0.0)
        self.data.qfrc_applied[:] = 1.234
        thermostat.apply(self.data)
        np.testing.assert_array_equal(self.data.qfrc_applied, 1.234)

    def test_apply_only_touches_its_own_dofs(self):
        """A caller's other generalized forces must survive the kick."""
        thermostat = self.thermostat(kT=0.05)
        others = np.setdiff1d(np.arange(self.model.nv), thermostat.dofs)
        self.data.qfrc_applied[others] = 7.0
        thermostat.apply(self.data)
        np.testing.assert_array_equal(self.data.qfrc_applied[others], 7.0)
        self.assertTrue(np.any(self.data.qfrc_applied[thermostat.dofs] != 0.0))

    def test_the_kick_is_overwritten_not_accumulated(self):
        """Repeated applies must not integrate into a runaway force."""
        thermostat = self.thermostat(kT=0.05)
        magnitudes = []
        for _ in range(200):
            thermostat.apply(self.data)
            magnitudes.append(np.abs(self.data.qfrc_applied[thermostat.dofs]).max())
        self.assertLess(max(magnitudes), 6.0 * thermostat.sigma.max())

    def test_seeds_are_reproducible_and_independent(self):
        """A sweep holds the scene fixed and varies only the noise."""
        draws = {}
        for seed in (0, 0, 1):
            data = mujoco.MjData(self.model)
            self.thermostat(kT=0.05, seed=seed).apply(data)
            draws.setdefault(seed, []).append(np.copy(data.qfrc_applied))
        np.testing.assert_array_equal(draws[0][0], draws[0][1])
        self.assertFalse(np.allclose(draws[0][0], draws[1][0]))

    def test_successive_draws_are_independent(self):
        """
        The stream runs continuously and is never reseeded, so two episodes from
        one engine see different noise rather than a replay of one realisation.
        """
        thermostat = self.thermostat(kT=0.05)
        thermostat.apply(self.data)
        first = np.copy(self.data.qfrc_applied)
        thermostat.apply(self.data)
        self.assertFalse(np.allclose(first, self.data.qfrc_applied))


@ut.skipUnless(MUJOCO_AVAILABLE, "mujoco is not installed")
class TestThermostatPhysics(ut.TestCase):
    """
    The two numbers that decide whether the thermostat is correct: the stationary
    velocity variance and the diffusion coefficient it produces.
    """

    kT = peclet_matched_kT(DRIVE, DAMPING, RADIUS) if MUJOCO_AVAILABLE else 0.0

    def test_equipartition_matches_the_discrete_time_prediction(self):
        """
        Measured <v^2> per DoF must match expected_variance.

        The comparison is against the implicit-damping fixed point rather than the
        continuum kT/m: MuJoCo's implicitfast integrator biases the former away from
        the latter, by ~1% at these parameters.
        """
        model = mujoco.MjModel.from_xml_string(free_particle_xml(DAMPING))
        data = mujoco.MjData(model)
        thermostat = LangevinThermostat(kT=self.kT, seed=3, planar=True)
        thermostat.bind(model, np.arange(1, N_WALKERS + 1), model.opt.timestep)

        mujoco.mj_forward(model, data)
        expected = thermostat.expected_variance(model, data)

        for _ in range(2000):  # let the velocities relax before sampling
            thermostat.apply(data)
            mujoco.mj_step(model, data)

        total = np.zeros(thermostat.dofs.size)
        n_samples = 20000
        for _ in range(n_samples):
            thermostat.apply(data)
            mujoco.mj_step(model, data)
            total += np.square(data.qvel[thermostat.dofs])
        ratio = (total / n_samples) / expected

        # Average across walkers before asserting. A translational DoF decorrelates
        # on tau_m = m/gamma = 167 steps, so a single one of them carries only ~120
        # independent samples here and scatters by ~13%; the 48-walker ensemble
        # brings that down to a couple of percent. Grouping by DoF kind (x, y, yaw)
        # also keeps the test able to see a systematically wrong axis.
        for offset, name in ((0, "x"), (1, "y"), (5, "yaw")):
            group = ratio[thermostat.dofs % 6 == offset]
            self.assertAlmostEqual(
                float(group.mean()),
                1.0,
                delta=0.05,
                msg=f"equipartition violated on {name}: {group.mean():.3f} x expected",
            )
        # And no single DoF may be dead or diverging.
        self.assertTrue(
            np.all(np.abs(ratio - 1.0) < 0.6),
            f"a DoF is far off equipartition: {np.round(ratio, 3)}",
        )

    def test_diffusion_coefficient_matches_kT_over_gamma(self):
        """
        Free-particle MSD must grow as 4 D t in the plane, with D = kT/gamma.

        The SLOPE of MSD(t) is fitted rather than dividing MSD by 4t. For a Langevin
        particle

            MSD(t) = 4 D [t - tau_m (1 - exp(-t/tau_m))],   tau_m = m/gamma

        which is a straight line of slope 4D offset by -4 D tau_m once t >> tau_m.
        The offset stops growing but never vanishes, so reading D off a single lag
        understates it and looks like a broken thermostat. The slope is unbiased.
        """
        damping = DAMPING
        model = mujoco.MjModel.from_xml_string(free_particle_xml(damping))
        data = mujoco.MjData(model)
        thermostat = LangevinThermostat(kT=self.kT, seed=5, planar=True)
        thermostat.bind(model, np.arange(1, N_WALKERS + 1), model.opt.timestep)

        dt = model.opt.timestep
        # qpos is [x y z qw qx qy qz] per walker; track the xy of each.
        xy = np.concatenate([[7 * i, 7 * i + 1] for i in range(N_WALKERS)])
        n_steps = 60000
        track = np.empty((n_steps, len(xy)))
        for k in range(n_steps):
            thermostat.apply(data)
            mujoco.mj_step(model, data)
            track[k] = data.qpos[xy]

        lag_seconds = np.array([1.5, 3.0, 4.5, 6.0])
        msds = []
        for seconds in lag_seconds:
            lag = int(seconds / dt)
            disp = track[lag:] - track[:-lag]
            msds.append(np.mean(disp**2) * 2.0)  # 2 dims, averaged over all walkers
        measured_D = float(np.polyfit(lag_seconds, msds, 1)[0] / 4.0)
        expected_D = self.kT / damping

        self.assertAlmostEqual(
            measured_D / expected_D,
            1.0,
            delta=0.1,
            msg=f"D = {measured_D:.6f}, expected kT/gamma = {expected_D:.6f}",
        )


if __name__ == "__main__":
    ut.main()
