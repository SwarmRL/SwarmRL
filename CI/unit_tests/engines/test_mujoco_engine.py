"""
Tests for the MuJoCo engine.

MuJoCo is an optional dependency, so the whole module skips when it is absent.
Nothing here renders: the physics needs no GL context, and a headless runner has
none.
"""

import typing
import unittest as ut

import numpy as np

from swarmrl.actions.actions import Action
from swarmrl.agents.classical_agent import ClassicalAgent
from swarmrl.components.colloid import Colloid
from swarmrl.force_functions import ForceFunction

try:
    import mujoco  # noqa: F401

    from swarmrl.engine.mujoco_engine import MujocoEngine, build_swarm_xml

    MUJOCO_AVAILABLE = True
except ModuleNotFoundError:
    MUJOCO_AVAILABLE = False


class ConstAgent(ClassicalAgent):
    """
    Agent that hands the same Action to every colloid of its type.

    ``dummy_models.ConstForce`` cannot be used here because the engine calls
    ``calc_reward`` once per slice and the classical agents do not implement it.
    """

    def __init__(self, particle_type: int, action: Action):
        super().__init__(particle_type=particle_type, actions={})
        self.action = action
        self.n_rewards = 0

    def calc_action(self, colloids: typing.List[Colloid]) -> typing.List[Action]:
        return [self.action for c in colloids if c.type == self.particle_type]

    def calc_reward(self, colloids, external_reward: float = 0.0) -> None:
        self.n_rewards += 1


def force_function(agents: dict) -> ForceFunction:
    """
    Wrap ``{particle_type: Action}`` in a real ForceFunction.
    """
    return ForceFunction(
        agents={str(t): ConstAgent(t, action) for t, action in agents.items()}
    )


@ut.skipUnless(MUJOCO_AVAILABLE, "mujoco is not installed")
class TestMujocoEngine(ut.TestCase):
    """
    Check that a MuJoCo model is faithfully exposed as SwarmRL Colloids and that
    Actions come back out as motion.
    """

    n_agents = 4

    def engine(self, **kwargs) -> "MujocoEngine":
        kwargs.setdefault("steps_per_slice", 25)
        return MujocoEngine(build_swarm_xml(self.n_agents, seed=1), **kwargs)

    def test_free_bodies_are_discovered(self):
        """Every free-jointed body becomes a Colloid, in model order."""
        engine = self.engine()
        self.assertEqual(
            engine.agent_bodies, [f"agent_{i}" for i in range(self.n_agents)]
        )
        self.assertEqual(len(engine.colloids), self.n_agents)
        self.assertEqual([c.id for c in engine.colloids], list(range(self.n_agents)))

    def test_colloids_mirror_the_mujoco_state(self):
        """Positions come from xpos and the directors are unit vectors."""
        engine = self.engine()
        positions = np.array([c.pos for c in engine.colloids])
        np.testing.assert_allclose(positions, engine.data.xpos[engine.body_ids])

        directors = np.array([c.director for c in engine.colloids])
        np.testing.assert_allclose(np.linalg.norm(directors, axis=1), 1.0, atol=1e-10)
        # build_swarm_xml only yaws the bodies, so every director stays in-plane.
        np.testing.assert_allclose(directors[:, 2], 0.0, atol=1e-10)

    def test_particle_types_are_broadcast(self):
        """A scalar type applies to all agents, a list applies element-wise."""
        self.assertEqual(self.engine().particle_types, [0] * self.n_agents)

        types = [0, 1, 1, 2]
        engine = self.engine(particle_types=types)
        self.assertEqual(engine.particle_types, types)
        self.assertEqual([c.type for c in engine.colloids], types)

    def test_unknown_body_raises(self):
        """A typo in agent_bodies must fail loudly rather than index -1."""
        with self.assertRaises(ValueError):
            MujocoEngine(build_swarm_xml(2), agent_bodies=["agent_0", "nope"])

    def test_model_without_free_joints_raises(self):
        """There is nothing to control, so refuse to build."""
        xml = """
        <mujoco>
          <worldbody><geom name="floor" type="plane" size="1 1 0.1"/></worldbody>
        </mujoco>
        """
        with self.assertRaises(ValueError):
            MujocoEngine(xml)

    def test_force_drives_the_agent_along_its_director(self):
        """A positive Action.force is propulsion along the heading."""
        engine = self.engine()
        directors = np.array([c.director for c in engine.colloids])
        start = np.copy(engine.data.xpos[engine.body_ids])

        engine.integrate(6, force_function({0: Action(force=20.0)}))

        displacement = engine.data.xpos[engine.body_ids] - start
        travelled = np.linalg.norm(displacement[:, :2], axis=1)
        self.assertTrue(np.all(travelled > 1e-3), f"agents did not move: {travelled}")

        alignment = np.einsum(
            "ni,ni->n", displacement[:, :2], directors[:, :2]
        ) / np.maximum(travelled, 1e-12)
        self.assertTrue(
            np.all(alignment > 0.95), f"motion is not along the director: {alignment}"
        )

    def test_planar_mode_keeps_the_system_two_dimensional(self):
        """z-forces are dropped and only the z component of a torque survives."""
        engine = self.engine()
        engine.manage_forces(
            force_function({0: Action(force=20.0, torque=np.array([3.0, 4.0, 5.0]))})
        )
        applied = engine.data.xfrc_applied[engine.body_ids]
        np.testing.assert_allclose(applied[:, 2], 0.0, atol=1e-12)
        np.testing.assert_allclose(applied[:, 3:5], 0.0, atol=1e-12)
        np.testing.assert_allclose(applied[:, 5], 5.0)

    def test_new_direction_sets_the_yaw(self):
        """The director follows new_direction rather than drifting toward it."""
        engine = self.engine()
        target = np.array([0.0, 1.0, 0.0])
        engine.manage_forces(force_function({0: Action(new_direction=target)}))

        directors = np.array([c.director for c in engine._build_colloids()])
        for director in directors:
            np.testing.assert_allclose(director, target, atol=1e-8)

    def test_reward_is_computed_once_per_slice(self):
        """The engine must drive the reward side of the ForceFunction too."""
        engine = self.engine()
        model = force_function({0: Action(force=1.0)})
        engine.integrate(5, model)
        self.assertEqual(model.agents["0"].n_rewards, 5)

    def test_kill_switch_stops_the_integration(self):
        """A model that asks to stop is obeyed before the first slice."""
        engine = self.engine()
        model = force_function({0: Action(force=20.0)})
        model.kill_switch = True
        engine.integrate(10, model)
        self.assertEqual(engine.slice_idx, 0)

    def test_trajectory_records_one_frame_per_slice(self):
        """record_traj gives the renderer a full qpos history."""
        engine = self.engine(record_traj=True)
        engine.integrate(7, force_function({0: Action(force=5.0)}))
        self.assertEqual(len(engine.traj["Times"]), 7)
        self.assertEqual(
            np.shape(engine.traj["Unwrapped_Positions"]), (7, self.n_agents, 3)
        )
        self.assertEqual(np.shape(engine.traj["qpos"]), (7, engine.model.nq))

    def test_get_particle_data_matches_the_colloids(self):
        """The trainer-facing view and the Colloid view must agree."""
        engine = self.engine(particle_types=[0, 1, 1, 2])
        engine.integrate(3, force_function({0: Action(force=5.0)}))
        data = engine.get_particle_data()

        np.testing.assert_array_equal(data["Id"], np.arange(self.n_agents))
        np.testing.assert_array_equal(data["Type"], [0, 1, 1, 2])
        np.testing.assert_allclose(
            data["Unwrapped_Positions"], [c.pos for c in engine.colloids]
        )
        np.testing.assert_allclose(
            data["Directors"], [c.director for c in engine.colloids]
        )

    def test_reset_restores_the_initial_state(self):
        """Episodic training reuses one engine, so reset has to be exact."""
        engine = self.engine()
        initial = np.copy(engine.data.qpos)

        engine.integrate(5, force_function({0: Action(force=20.0)}))
        self.assertFalse(np.allclose(engine.data.qpos, initial))

        engine.reset()
        np.testing.assert_allclose(engine.data.qpos, initial)
        self.assertEqual(engine.slice_idx, 0)
        self.assertEqual(len(engine.traj["Times"]), 0)

    def test_two_species_get_their_own_actions(self):
        """Types are routed independently, as they are in the espresso engine."""
        engine = self.engine(particle_types=[0, 0, 1, 1])
        start = np.copy(engine.data.xpos[engine.body_ids])
        engine.integrate(
            6, force_function({0: Action(force=20.0), 1: Action(force=0.0)})
        )
        travelled = np.linalg.norm(
            (engine.data.xpos[engine.body_ids] - start)[:, :2], axis=1
        )
        self.assertTrue(
            np.all(travelled[:2] > 1e-3), f"driven agents idle: {travelled}"
        )
        self.assertTrue(np.all(travelled[2:] < travelled[:2].min()))


if __name__ == "__main__":
    ut.main()
