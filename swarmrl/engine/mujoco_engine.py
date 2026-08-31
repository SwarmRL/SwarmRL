"""
MuJoCo engine for SwarmRL.

Bridges a MuJoCo model to the SwarmRL Engine interface: bodies with free joints are
exposed as Colloids, and the Actions returned by the ForceFunction are applied as
Cartesian forces/torques via xfrc_applied.

MuJoCo is an optional dependency: install it with ``pip install "swarmrl[mujoco]"``.
Trajectory rendering additionally needs a working GL backend, which on a headless
machine means ``MUJOCO_GL=egl``.
"""

import typing

import numpy as np
from loguru import logger

from swarmrl.components.colloid import Colloid
from swarmrl.force_functions import ForceFunction

from .engine import Engine
from .mujoco_thermostat import LangevinThermostat  # noqa: F401  (type reference)

try:
    import mujoco
except ModuleNotFoundError:
    logger.warning("Could not find mujoco. Features will not be available")


def build_swarm_xml(
    n_agents: int,
    radius: float = 0.05,
    height: float = 0.02,
    arena_size: float = 2.0,
    spawn_radius: float = 0.5,
    damping: float = 2.0,
    seed: int = 42,
) -> str:
    """
    Build an MJCF describing n_agents pucks on a frictional plane.

    The pucks are cylinders with free joints, so they collide with each other and
    with the arena walls. Damping is applied through the free joints, which puts the
    system in the overdamped regime that self-propelled agents usually live in.

    Parameters
    ----------
    n_agents : int
            Number of agents to place in the arena.
    radius : float
            Radius of a single puck.
    height : float
            Half-height of a single puck.
    arena_size : float
            Half-width of the square arena.
    spawn_radius : float
            Agents are placed uniformly in a disc of this radius.
    damping : float
            Free-joint damping. MuJoCo applies one value to all six DoFs, so this
            damps translation and rotation alike.
    seed : int
            Seed for the initial placement.

    Returns
    -------
    xml : str
            MJCF string ready for MjModel.from_xml_string.
    """
    rng = np.random.default_rng(seed)
    # Rejection-free placement: sample on a jittered ring so pucks never start
    # overlapping, which MuJoCo would resolve with a large initial impulse.
    angles = rng.permutation(np.linspace(0, 2 * np.pi, n_agents, endpoint=False))
    radii = spawn_radius * np.sqrt(rng.uniform(0.15, 1.0, size=n_agents))

    bodies = []
    for i in range(n_agents):
        x = radii[i] * np.cos(angles[i])
        y = radii[i] * np.sin(angles[i])
        yaw = rng.uniform(0, 2 * np.pi)
        quat = f"{np.cos(yaw / 2)} 0 0 {np.sin(yaw / 2)}"
        bodies.append(f"""
    <body name="agent_{i}" pos="{x:.4f} {y:.4f} {height}" quat="{quat}">
      <freejoint name="agent_{i}_joint"/>
      <geom name="agent_{i}_geom" type="cylinder" size="{radius} {height}"
            rgba="0.2 0.5 0.9 1" friction="0.4 0.005 0.0001"/>
      <!-- Marker along local +x so the director is visible when rendering. -->
      <geom type="box" size="{radius * 0.9} {radius * 0.18} {height * 1.1}"
            pos="{radius * 0.5} 0 0" rgba="1 0.4 0.1 1" contype="0" conaffinity="0"
            mass="0"/>
    </body>""")

    wall_h = 4 * height
    walls = []
    for name, pos, size in [
        ("wall_px", f"{arena_size} 0 {wall_h}", f"0.02 {arena_size} {wall_h}"),
        ("wall_nx", f"{-arena_size} 0 {wall_h}", f"0.02 {arena_size} {wall_h}"),
        ("wall_py", f"0 {arena_size} {wall_h}", f"{arena_size} 0.02 {wall_h}"),
        ("wall_ny", f"0 {-arena_size} {wall_h}", f"{arena_size} 0.02 {wall_h}"),
    ]:
        walls.append(
            f'<geom name="{name}" type="box" pos="{pos}" size="{size}"'
            ' rgba="0.6 0.6 0.6 0.3"/>'
        )

    return f"""
<mujoco model="swarm">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <joint damping="{damping}"/>
    <geom solref="0.005 1"/>
  </default>
  <visual>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"
           specular="0.1 0.1 0.1" ambient="0.45 0.45 0.45"/>
    <camera name="top" pos="0 0 {3.2 * arena_size}" quat="1 0 0 0"/>
    <geom name="floor" type="plane" size="{arena_size} {arena_size} 0.1"
          rgba="0.9 0.9 0.92 1" friction="0.4 0.005 0.0001"/>
    {"".join(walls)}
    {"".join(bodies)}
  </worldbody>
</mujoco>
"""


class MujocoEngine(Engine):
    """
    SwarmRL Engine backed by MuJoCo.

    Every body named in ``agent_bodies`` is exposed to SwarmRL as a Colloid. Actions
    are applied as Cartesian force/torque through ``data.xfrc_applied``, so the model
    itself needs no actuators.

    Attributes
    ----------
    model : mujoco.MjModel
            The compiled MuJoCo model.
    data : mujoco.MjData
            The MuJoCo state.
    colloids : list of Colloid
            Current SwarmRL view of the agents.
    """

    def __init__(
        self,
        xml: str,
        agent_bodies: typing.List[str] = None,
        particle_types: typing.Union[int, typing.List[int]] = 0,
        steps_per_slice: int = 25,
        director_axis: typing.Sequence[float] = (1.0, 0.0, 0.0),
        planar: bool = True,
        record_traj: bool = True,
        seed: int = 42,
        thermostat: "LangevinThermostat" = None,
    ):
        """
        Parameters
        ----------
        xml : str
                MJCF string, or a path to an .xml file.
        agent_bodies : list of str
                Names of the bodies to expose as agents. If None, every body carrying
                a free joint is used, in model order.
        particle_types : int or list of int
                SwarmRL particle type per agent. A single int is broadcast. These must
                match the ``particle_type`` of the agents handed to the Trainer.
        steps_per_slice : int
                MuJoCo steps between two action updates. One "slice" is one RL step.
        director_axis : sequence of float
                Body-local axis that defines the director (heading).
        planar : bool
                If True, ``new_direction`` is applied as a rotation about z only, and
                z-forces are suppressed. This keeps a 2D system 2D.
        record_traj : bool
                If True, positions/directors are stored each slice in ``self.traj``.
        seed : int
                Seed for MuJoCo's internal RNG (used by noise/randomization).
        thermostat : LangevinThermostat
                If given, a fresh Langevin kick is applied to every thermostatted
                degree of freedom before each MuJoCo step. None leaves the
                system athermal.
        """
        if xml.strip().startswith("<"):
            self.model = mujoco.MjModel.from_xml_string(xml)
        else:
            self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)

        self.steps_per_slice = steps_per_slice
        self.planar = planar
        self.record_traj = record_traj
        self.seed = seed

        if agent_bodies is None:
            agent_bodies = self._find_free_bodies()
        if len(agent_bodies) == 0:
            raise ValueError("No agent bodies found; the model has no free joints.")
        self.agent_bodies = list(agent_bodies)

        self.body_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.agent_bodies
        ])
        if np.any(self.body_ids < 0):
            missing = [n for n, i in zip(self.agent_bodies, self.body_ids) if i < 0]
            raise ValueError(f"Bodies not found in the model: {missing}")

        # qpos address of each agent's first joint. For a free joint this is
        # [x y z qw qx qy qz]; for a hinge it is the single joint angle.
        jnt_adr = np.array([self.model.body_jntadr[b] for b in self.body_ids])
        self.qpos_adr = np.array([self.model.jnt_qposadr[a] for a in jnt_adr])
        self.qvel_adr = np.array([self.model.jnt_dofadr[a] for a in jnt_adr])
        # Rail-mounted agents (hinge joints) cannot be reoriented freely, so
        # new_direction is ignored for them.
        self.is_free = np.array([
            self.model.jnt_type[a] == mujoco.mjtJoint.mjJNT_FREE for a in jnt_adr
        ])

        # The director axis is body-local and may differ per agent: a free swimmer
        # usually points along its local +x, while a hinge-mounted wall walker moves
        # along its local +y (the tangent).
        axes = np.asarray(director_axis, dtype=float)
        if axes.ndim == 1:
            axes = np.tile(axes, (len(self.body_ids), 1))
        self.director_axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)

        if np.isscalar(particle_types):
            self.particle_types = [int(particle_types)] * len(self.agent_bodies)
        else:
            self.particle_types = [int(t) for t in particle_types]

        self.slice_idx = 0
        self.traj = {
            "Times": [],
            "Unwrapped_Positions": [],
            "Directors": [],
            "qpos": [],
        }

        self.thermostat = thermostat
        if self.thermostat is not None:
            self.thermostat.bind(self.model, self.body_ids, self.model.opt.timestep)

        mujoco.mj_forward(self.model, self.data)
        self.colloids = self._build_colloids()

    def _find_free_bodies(self) -> typing.List[str]:
        """
        Return the names of all bodies whose first joint is a free joint.
        """
        names = []
        for b in range(1, self.model.nbody):  # body 0 is the world
            adr = self.model.body_jntadr[b]
            if adr < 0 or self.model.body_jntnum[b] == 0:
                continue
            if self.model.jnt_type[adr] == mujoco.mjtJoint.mjJNT_FREE:
                names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b))
        return names

    def _directors(self) -> np.ndarray:
        """
        Director of every agent in world coordinates, shape (n_agents, 3).
        """
        mats = self.data.xmat[self.body_ids].reshape(-1, 3, 3)
        return np.einsum("nij,nj->ni", mats, self.director_axes)

    def _velocities(self) -> np.ndarray:
        """
        World-frame linear velocity of every agent, shape (n_agents, 3).
        """
        vel = np.empty((len(self.body_ids), 6))
        for i, bid in enumerate(self.body_ids):
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, bid, vel[i], 0
            )
        return vel[:, 3:]  # [angular(3), linear(3)] -> linear part

    def _build_colloids(self) -> typing.List[Colloid]:
        """
        Snapshot the MuJoCo state as a list of SwarmRL Colloids.
        """
        positions = self.data.xpos[self.body_ids]
        directors = self._directors()
        velocities = self._velocities()
        return [
            Colloid(
                pos=np.copy(positions[i]),
                director=np.copy(directors[i]),
                id=i,
                velocity=np.copy(velocities[i]),
                type=self.particle_types[i],
            )
            for i in range(len(self.body_ids))
        ]

    def _set_director(self, index: int, new_direction: np.ndarray):
        """
        Rotate an agent so that its director points along new_direction.

        In planar mode only the yaw is changed, which keeps the puck flat on the
        floor; otherwise the shortest rotation from the current director is applied.
        Rail-mounted (hinge) agents have no free orientation and are left alone.
        """
        if not self.is_free[index]:
            return

        adr = self.qpos_adr[index]
        new_direction = np.asarray(new_direction, dtype=float)

        if self.planar:
            yaw = np.arctan2(new_direction[1], new_direction[0])
            self.data.qpos[adr + 3 : adr + 7] = [
                np.cos(yaw / 2),
                0.0,
                0.0,
                np.sin(yaw / 2),
            ]
            return

        current = self._directors()[index]
        new_direction = new_direction / np.linalg.norm(new_direction)
        axis = np.cross(current, new_direction)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            return
        angle = np.arctan2(axis_norm, np.dot(current, new_direction))
        delta = np.empty(4)
        mujoco.mju_axisAngle2Quat(delta, axis / axis_norm, angle)
        result = np.empty(4)
        mujoco.mju_mulQuat(result, delta, self.data.qpos[adr + 3 : adr + 7])
        self.data.qpos[adr + 3 : adr + 7] = result

    def manage_forces(self, force_model: ForceFunction = None):
        """
        Query the force model and write the resulting actions into MuJoCo.

        Parameters
        ----------
        force_model : ForceFunction
                Model with which to compute the actions.
        """
        self.data.xfrc_applied[:] = 0.0
        if force_model is None:
            return

        self.colloids = self._build_colloids()
        actions = force_model.calc_action(self.colloids)

        directors = self._directors()
        reorient = []
        for i, action in enumerate(actions):
            bid = self.body_ids[i]

            if action.force:
                f = float(action.force) * directors[i]
                if self.planar:
                    f[2] = 0.0
                self.data.xfrc_applied[bid, :3] = f

            if action.torque is not None:
                torque = np.asarray(action.torque, dtype=float)
                if self.planar:
                    torque = np.array([0.0, 0.0, torque[2]])
                self.data.xfrc_applied[bid, 3:] = torque

            if action.new_direction is not None:
                reorient.append((i, action.new_direction))

        # Writing qpos invalidates the derived quantities, so recompute once after
        # all reorientations rather than per agent.
        if reorient:
            for i, new_direction in reorient:
                self._set_director(i, new_direction)
            mujoco.mj_forward(self.model, self.data)

    def integrate(self, n_slices: int, force_model: ForceFunction = None) -> None:
        """
        Integrate the system for n_slices RL steps.

        Parameters
        ----------
        n_slices : int
                Number of action updates to perform.
        force_model : ForceFunction
                A SwarmRL interaction model deciding the agent actions.
        """
        for _ in range(int(n_slices)):
            if force_model is not None and force_model.kill_switch:
                break

            self._pre_slice()
            self.manage_forces(force_model)

            for _ in range(self.steps_per_slice):
                self._pre_step()
                mujoco.mj_step(self.model, self.data)

            self.colloids = self._build_colloids()
            # Environment state maintained by a subclass must be current
            # before the task computes its reward.
            self._post_step()

            if force_model is not None:
                force_model.calc_reward(self.colloids)

            if self.record_traj:
                self.traj["Times"].append(self.data.time)
                self.traj["Unwrapped_Positions"].append(
                    np.copy(self.data.xpos[self.body_ids])
                )
                self.traj["Directors"].append(self._directors())
                # Full qpos lets the renderer replay any joint type, including
                # hinge-mounted agents and non-agent bodies.
                self.traj["qpos"].append(np.copy(self.data.qpos))

            self.slice_idx += 1

    def _pre_slice(self):
        """
        Hook run before actions are computed. Overridden by subclasses.
        """
        pass

    def _pre_step(self):
        """
        Hook run before every MuJoCo step, i.e. steps_per_slice times per slice.

        Applies the thermostat. Unlike the policy action, which is written once per
        slice, thermal noise has to be redrawn every step to stay white.
        """
        if self.thermostat is not None:
            self.thermostat.apply(self.data)

    def _post_step(self):
        """
        Hook run after integration, before rewards. Overridden by subclasses.
        """
        pass

    def get_particle_data(self) -> dict:
        """
        Get position, velocity, director and type of the agents.

        Returns
        -------
        data : dict
                Dict of np.ndarray, one row per agent.
        """
        return {
            "Id": np.arange(len(self.body_ids)),
            "Type": np.array(self.particle_types),
            "Unwrapped_Positions": np.copy(self.data.xpos[self.body_ids]),
            "Velocities": self._velocities(),
            "Directors": self._directors(),
        }

    def reset(self):
        """
        Reset the simulation to the model's initial state.
        """
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.slice_idx = 0
        self.traj = {
            "Times": [],
            "Unwrapped_Positions": [],
            "Directors": [],
            "qpos": [],
        }
        self.colloids = self._build_colloids()

    def render_trajectory(
        self,
        out_path: str,
        width: int = 640,
        height: int = 480,
        camera: str = "top",
        stride: int = 4,
        fps: int = 30,
    ):
        """
        Replay the recorded trajectory offscreen and write it to an mp4/gif.

        Requires MUJOCO_GL to be set to a working backend (egl when headless) and
        imageio to be installed.

        Parameters
        ----------
        out_path : str
                Destination file. The extension selects the format.
        width, height : int
                Frame size in pixels.
        camera : str
                Name of the camera defined in the MJCF.
        stride : int
                Render every stride-th recorded slice.
        fps : int
                Frame rate of the output file.
        """
        import imageio.v3 as iio

        if not self.traj["Times"]:
            raise RuntimeError("No trajectory recorded; construct with record_traj.")

        scratch = mujoco.MjData(self.model)
        frames = []
        with mujoco.Renderer(self.model, height, width) as renderer:
            for k in range(0, len(self.traj["Times"]), stride):
                # Replaying the full qpos reproduces every body exactly, including
                # hinge-mounted agents and passive objects.
                scratch.qpos[:] = self.traj["qpos"][k]
                mujoco.mj_forward(self.model, scratch)
                renderer.update_scene(scratch, camera=camera)
                frames.append(renderer.render())

        iio.imwrite(out_path, np.stack(frames), fps=fps)
        return out_path

    def finalize(self):
        """
        Nothing to flush; kept for interface compatibility.
        """
        pass
