import copy
import multiprocessing
import os
import pickle
import tempfile
import time
import traceback
import typing
import unittest as ut

import matplotlib.pyplot as plt
import numpy as np
import pint
import tqdm
from matplotlib.animation import FuncAnimation
from scipy.special import kv, kvp

import swarmrl.engine.espresso as espresso
from swarmrl.models.interaction_model import Action, InteractionModel
from swarmrl.observables.subdivided_vision_cones import SubdividedVisionCones
from swarmrl.utils import utils
from swarmrl.visualization.video_vis import (
    Animations,
    load_extra_data_to_visualization,
    load_traj_vis,
)


class Process(multiprocessing.Process):
    """
    Process class for use in multi simulation testing.
    """

    def __init__(self, *args, **kwargs):
        """
        Multiprocessing class constructor.
        """
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        """
        Run the process and catch exceptions.
        """
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self):
        """
        Exception property to be stored by the process.
        """
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def calc_schmell(
    schmellpos, colpos, diffusion_col_trans=670
):  # glucose in water in mum^2/s
    delta_distance = np.linalg.norm(np.array(schmellpos) - np.array(colpos), axis=-1)
    delta_distance = np.where(delta_distance == 0, 5, delta_distance)
    # prevent zero division
    direction = (schmellpos - colpos) / np.stack(
        [delta_distance, delta_distance], axis=-1
    )

    Out = 0.4  # in per second
    J_flow = 1800  # in chemics per second whatever units the chemics are in.
    # These values are chosen acording to the Amplitude \approx 1
    # const = -2 * np.pi * diffusion_col_trans * rod_thickness / 2 *
    # np.sqrt(Out / diffusion_col_trans) * kvp(0, np.sqrt(
    #   Out / diffusion_col_trans) * rod_thickness / 2)
    #   reuse already calculated value
    const = 4182.925639571625
    # J_flow=const*A the const sums over the chemics that
    # flow throw an imaginary boundary at radius rod_thickness/2
    # A is the Amplitude of the potential
    # i.e. the chemical density (google "first ficks law" for theoretical info)
    Amplitude = J_flow / const

    length = np.sqrt(Out / diffusion_col_trans) * delta_distance
    schmell_magnitude = Amplitude * kv(0, length)
    schmell_gradient = (
        -np.stack(
            [
                Amplitude * np.sqrt(Out / diffusion_col_trans) * kvp(0, length),
                Amplitude * np.sqrt(Out / diffusion_col_trans) * kvp(0, length),
            ],
            axis=-1,
        )
        * direction
    )
    return schmell_magnitude, schmell_gradient


class minimum_model(InteractionModel):
    def __init__(
        self,
        act_force=42,
        act_torque=42,
        n_type=[],
        center_point=[500, 500],
        phase_len=[4, 4],
        acts_on_types: typing.List[int] = None,
    ):
        self.act_force = act_force
        self.act_torque = act_torque
        self.n_type = n_type
        self.center_point = center_point
        self.phase_len = phase_len
        if acts_on_types is None:
            acts_on_types = [0]
        self.acts_on_types = acts_on_types

        self.n_col = sum(self.n_type)
        # the colloids individually keep track in which state they are
        # this is stored in col_RAM (Random Access Memory) alongside with additional
        # space for memory, all of it yield to a action decision
        self.n_phases = 2  # run, steer
        self.n_memory = 2  # schmellmemory
        self.col_RAM = np.zeros((self.n_col, self.n_phases + self.n_memory))
        self.col_RAM[:, 0] = 1  # starting in phase 0 with a single step

        # if not propagated manually the phases translate automatically
        # in this case from 0 to 1 and 1 to 0 according to the "Tupelschreibweise"
        # off permutations
        self.phase_trans = [1, 0]

    def calc_action(self, colloids) -> typing.List[Action]:
        actions = []

        for colloid in colloids:
            if colloid.type not in self.acts_on_types:
                actions.append(Action())
                continue

            # setup phases make sure to be in one
            if sum(self.col_RAM[colloid.id, : self.n_phases]) == 0:
                phase = 0
            else:
                ([phase],) = np.where(self.col_RAM[colloid.id, : self.n_phases] != 0)

            schmell_magnitude, _ = calc_schmell(self.center_point, colloid.pos[:2])

            # make decisions
            if phase == 0 and self.col_RAM[colloid.id, phase] == self.phase_len[phase]:
                # self.n_phase is the first index for schmellmemory
                if self.col_RAM[colloid.id, self.n_phases] < schmell_magnitude:
                    # if the chemics (schmell) get more
                    self.col_RAM[colloid.id, phase] = 0
                    phase = 0  # jump to run phase and head straight forward
                    self.col_RAM[colloid.id, phase] = 1
                else:
                    self.col_RAM[colloid.id, phase] = 0
                    phase = 1  # jump to steer phase
                    self.col_RAM[colloid.id, phase] = 1
                    # pass
                self.col_RAM[colloid.id, self.n_phases] = schmell_magnitude

            # make actions from decisions
            if phase == 0:  # run
                force_mult = 1
                torque_mult = 0
            elif phase == 1:  # steer
                force_mult = 0
                # depending on type turn a different direction
                torque_mult = 1 * ((colloid.type % 2) * 2 - 1)
            else:
                raise Exception(
                    "Colloid doesn't know what to do. Unexpected phase identifier"
                    " selected"
                )

            actions.append(
                Action(
                    force=self.act_force * force_mult,
                    torque=np.array([0, 0, self.act_torque * torque_mult]),
                )
            )

            # propagate phases automatically
            for j in range(self.n_phases):
                if phase == j and self.col_RAM[colloid.id, phase] == self.phase_len[j]:
                    self.col_RAM[colloid.id, phase] = 0
                    phase = self.phase_trans[j]
            self.col_RAM[colloid.id, phase] += 1
        return actions


class extra_data_model(InteractionModel):
    def __init__(
        self,
        act_force=42,
        act_torque=42,
        n_type=[],
        center_point=[500, 500],
        phase_len=[4, 4],
        data_folder="example_folder",
        acts_on_types: typing.List[int] = None,
    ):
        self.act_force = act_force
        self.act_torque = act_torque
        self.n_type = n_type
        self.center_point = center_point
        self.phase_len = phase_len
        if acts_on_types is None:
            acts_on_types = [0]
        self.acts_on_types = acts_on_types

        self.n_col = sum(self.n_type)
        # the colloids individually keep track in which state they are
        # this is stored in col_RAM (Random Access Memory) alongside with additional
        # space for memory, all of it yield to a action decision
        self.n_phases = 2  # run, steer
        self.n_memory = 2  # schmellmemory
        self.col_RAM = np.zeros((self.n_col, self.n_phases + self.n_memory))
        self.col_RAM[:, 0] = 1  # starting in phase 0 with a single step

        # if not propagated manually the phases translate automatically
        # in this case from 0 to 1 and 1 to 0 according to the "Tupelschreibweise"
        # off permutations
        self.phase_trans = [1, 0]

        self.vision_cone_data = []
        self.written_info_data = None
        files = os.listdir(data_folder)
        if "written_info_data.txt" in files:
            os.remove(data_folder + "/written_info_data.txt")
        self.written_info_data_file = open(data_folder + "/written_info_data.txt", "a")
        if "vision_cone_data.pick" in files:
            os.remove(data_folder + "/vision_cone_data.pick")
        self.vision_cone_data_file = open(data_folder + "/vision_cone_data.pick", "ab")

        self.vision_handle = SubdividedVisionCones(50, np.pi / 2, 5, [2.5] * self.n_col)

    def close_written_info_data_file(self):
        self.written_info_data_file.close()

    def close_vision_cone_data_file(self):
        self.vision_cone_data_file.close()

    def calc_action(self, colloids) -> typing.List[Action]:
        actions = []

        cone_data = []

        for colloid in colloids:
            if colloid.type not in self.acts_on_types:
                actions.append(Action())
                continue

            # setup phases make sure to be in one
            if sum(self.col_RAM[colloid.id, : self.n_phases]) == 0:
                phase = 0
            else:
                ([phase],) = np.where(self.col_RAM[colloid.id, : self.n_phases] != 0)

            vision_vals = self.vision_handle.compute_single_observable(
                colloid.id, colloids
            )
            cone_data.append([colloid.id, vision_vals])

            if colloid.id == 0:
                written_info_string = (
                    "green "
                    + str(np.round(np.array(vision_vals[:, 1]) * 100, 0))
                    + "*e-2"
                )
                self.written_info_data_file.write(written_info_string + "\n")

            schmell_magnitude, _ = calc_schmell(self.center_point[:2], colloid.pos[:2])

            # make decisions
            if phase == 0 and self.col_RAM[colloid.id, phase] == self.phase_len[phase]:
                if (
                    self.col_RAM[colloid.id, self.n_phases] < schmell_magnitude
                ):  # self.n_phase is the first index for schmellmemory
                    # if the chemics (schmell) get more
                    self.col_RAM[colloid.id, phase] = 0
                    phase = 0  # jump to run phase and head straight forward
                    self.col_RAM[colloid.id, phase] = 1
                else:
                    self.col_RAM[colloid.id, phase] = 0
                    phase = 1  # jump to steer phase
                    self.col_RAM[colloid.id, phase] = 1
                    # pass
                self.col_RAM[colloid.id, self.n_phases] = schmell_magnitude

            # make actions from decisions
            if phase == 0:  # run
                force_mult = 1
                torque_mult = 0
            elif phase == 1:  # steer
                force_mult = 0
                torque_mult = 1
            else:
                raise Exception(
                    "Colloid doesn't know what to do. Unexpected phase identifier"
                    " selected"
                )

            actions.append(
                Action(
                    force=self.act_force * force_mult,
                    torque=np.array([0, 0, self.act_torque * torque_mult]),
                )
            )

            # propagate phases automatically
            for j in range(self.n_phases):
                if phase == j and self.col_RAM[colloid.id, phase] == self.phase_len[j]:
                    self.col_RAM[colloid.id, phase] = 0
                    phase = self.phase_trans[j]
            self.col_RAM[colloid.id, phase] += 1

        pickle.dump(cone_data, self.vision_cone_data_file)

        return actions


class Simulation:
    def __init__(
        self,
        outfolder,
        simulation_name,
        loglevel_terminal,
        seed,
        mode,
    ):
        self.outfolder = outfolder
        self.simulation_name = simulation_name
        self.loglevel_terminal = loglevel_terminal
        self.seed = seed
        self.mode = mode

    def simulate_model(self):
        logger = utils.setup_swarmrl_logger(
            f"{self.outfolder}/{self.simulation_name}.log",
            loglevel_terminal=self.loglevel_terminal,
        )
        logger.info("Starting simulation setup")

        ureg = pint.UnitRegistry()
        md_params = espresso.MDParams(
            ureg=ureg,
            fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
            WCA_epsilon=ureg.Quantity(293, "kelvin") * ureg.boltzmann_constant,
            temperature=ureg.Quantity(293, "kelvin"),
            box_length=ureg.Quantity(1000, "micrometer"),
            time_slice=ureg.Quantity(0.2, "second"),  # model timestep
            time_step=ureg.Quantity(0.2, "second") / 5,  # integrator timestep
            write_interval=ureg.Quantity(2, "second"),
        )

        # parameters needed for the force models
        col_params = {
            "col_particle_type": 0,
            "target_vel_SI": ureg.Quantity(0.5, "micrometer / second"),
            "target_ang_vel_SI": ureg.Quantity(4 * np.pi / 180, "1/second"),
            "col_radius": ureg.Quantity(3.15, "micrometer"),
        }

        run_params = {
            "n_colloids": 10,
            "sim_duration": ureg.Quantity(200, "second"),
            "seed": self.seed,
            "write_interval_fac": int(md_params.write_interval / md_params.time_slice),
        }

        rod_params = {
            "rod_particle_type": 1,
            "rod_center": ureg.Quantity([500, 500, 0], "micrometer"),
            "rod_length": ureg.Quantity(100, "micrometer"),
            "rod_thickness": ureg.Quantity(100 / 59, "micrometer"),
            "rod_start_angle": 0,
            "n_particles": 59,
            "friction_trans": ureg.Quantity(
                4.388999692570726e-07, "newton * second / meter"
            ),
            "friction_rot": ureg.Quantity(
                6.902407326480637e-16, "meter * newton * second"
            ),
            "rod_fixed": True,
        }
        maze_params = {
            "maze_particle_type": 2,
            "maze_dic": {
                "x_col_pos": [
                    510.0,
                    510.0,
                    510.0,
                    550.0,
                    570.0,
                    550.0,
                    550.0,
                    570.0,
                    510.0,
                    570.0,
                    570.0,
                    530.0,
                    530.0,
                    530.0,
                    550.0,
                ],
                "y_col_pos": [
                    530.0,
                    550.0,
                    570.0,
                    510.0,
                    510.0,
                    570.0,
                    550.0,
                    550.0,
                    510.0,
                    530.0,
                    570.0,
                    510.0,
                    530.0,
                    570.0,
                    530.0,
                ],
                "destination": [530.0, 550.0],
                "offset": [500, 500],
                "wall_thickness": 2,
            },
            "maze_walls": [
                [500.0, 500.0, 500.0, 520.0],
                [500.0, 520.0, 500.0, 540.0],
                [500.0, 540.0, 500.0, 560.0],
                [500.0, 560.0, 500.0, 580.0],
                [500.0, 500.0, 520.0, 500.0],
                [500.0, 560.0, 520.0, 560.0],
                [500.0, 580.0, 520.0, 580.0],
                [520.0, 500.0, 520.0, 520.0],
                [520.0, 520.0, 520.0, 540.0],
                [520.0, 500.0, 540.0, 500.0],
                [520.0, 540.0, 540.0, 540.0],
                [520.0, 580.0, 540.0, 580.0],
                [540.0, 520.0, 540.0, 540.0],
                [540.0, 540.0, 540.0, 560.0],
                [540.0, 500.0, 560.0, 500.0],
                [540.0, 540.0, 560.0, 540.0],
                [540.0, 580.0, 560.0, 580.0],
                [560.0, 560.0, 560.0, 580.0],
                [560.0, 500.0, 580.0, 500.0],
                [560.0, 520.0, 580.0, 520.0],
                [560.0, 580.0, 580.0, 580.0],
                [580.0, 500.0, 580.0, 520.0],
                [580.0, 520.0, 580.0, 540.0],
                [580.0, 540.0, 580.0, 560.0],
                [580.0, 560.0, 580.0, 580.0],
            ],
        }

        # from now on, no new parameters are introduced

        system_runner = espresso.EspressoMD(
            md_params=md_params,
            n_dims=2,
            seed=run_params["seed"],
            out_folder=self.outfolder,
            write_chunk_size=1000,
        )

        if self.mode in ["minimum", "rod"]:
            logger.info(
                "Add " + str(run_params["n_colloids"]) + " colloids to simulation"
            )
            coll_type = 0
            system_runner.add_colloids(
                run_params["n_colloids"],
                col_params["col_radius"],
                ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
                ureg.Quantity(60, "micrometer"),
                type_colloid=coll_type,
            )

        if self.mode == "multiple_types":
            logger.info(
                "Add "
                + str(run_params["n_colloids"])
                + " colloids type 0 and also 1 to simulation"
            )
            # friction coefficients for both types are the same
            coll_type = 0
            system_runner.add_colloids(
                run_params["n_colloids"],
                col_params["col_radius"],
                ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
                ureg.Quantity(60, "micrometer"),
                type_colloid=coll_type,
            )
            system_runner.add_colloids(
                run_params["n_colloids"],
                col_params["col_radius"],
                ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
                ureg.Quantity(60, "micrometer"),
                type_colloid=1,
            )

        if self.mode == "rod":
            logger.info("Add rod to simulation")
            system_runner.add_rod(
                rod_params["rod_center"],
                rod_params["rod_length"],
                rod_params["rod_thickness"],
                rod_params["rod_start_angle"],
                rod_params["n_particles"],
                rod_params["friction_trans"],
                rod_params["friction_rot"],
                rod_params["rod_particle_type"],
                rod_params["rod_fixed"],
            )

        if self.mode == "maze":
            coll_type = 0
            maze_dic = maze_params["maze_dic"]
            logger.info("Add colloids in maze to simulation")
            run_params["n_colloids"] = len(maze_dic["x_col_pos"])
            for col_i in range(len(maze_dic["x_col_pos"])):
                system_runner.add_colloid_precisely(
                    radius_colloid=col_params["col_radius"],
                    init_position=ureg.Quantity(
                        [maze_dic["x_col_pos"][col_i], maze_dic["y_col_pos"][col_i], 0],
                        "micrometer",
                    ),
                    init_2D_angle=2 * np.pi * np.random.rand(),
                    type_colloid=coll_type,
                )
            system_runner.add_maze(
                maze_params["maze_walls"],
                maze_params["maze_particle_type"],
                maze_dic["wall_thickness"],
            )

        gamma, gamma_rot = system_runner.get_friction_coefficients(coll_type)
        target_vel = col_params["target_vel_SI"].m_as("sim_velocity")
        act_force = target_vel * gamma
        target_ang_vel = col_params["target_ang_vel_SI"].m_as("1 / sim_time")
        act_torque = target_ang_vel * gamma_rot

        if self.mode == "minimum":
            force_model = minimum_model(
                act_force=act_force,
                act_torque=act_torque,
                n_type=[run_params["n_colloids"], rod_params["n_particles"]],
                center_point=[500, 500],
                phase_len=[4, 4],
            )
        if self.mode == "multiple_types":
            force_model = extra_data_model(
                act_force=act_force,
                act_torque=act_torque,
                n_type=[run_params["n_colloids"], run_params["n_colloids"]],
                center_point=[500, 500],
                phase_len=[4, 4],
                acts_on_types=[0, 1],
                data_folder=self.outfolder,
            )
        if self.mode == "rod":
            force_model = extra_data_model(
                act_force=act_force,
                act_torque=act_torque,
                n_type=[run_params["n_colloids"], rod_params["n_particles"]],
                center_point=rod_params["rod_center"].magnitude,
                phase_len=[4, 4],
                data_folder=self.outfolder,
            )
        if self.mode == "maze":
            force_model = minimum_model(
                act_force=act_force,
                act_torque=act_torque,
                n_type=[run_params["n_colloids"]],
                center_point=[550, 550],
                phase_len=[4, 4],
            )

        n_slices = int(np.ceil(run_params["sim_duration"] / md_params.time_slice))
        run_params["n_slices"] = n_slices

        logger.info("Starting simulation")

        for _ in tqdm.tqdm(range(100)):
            system_runner.integrate(int(np.ceil(n_slices / 100)), force_model)

        system_runner.finalize()
        logger.info("Simulation completed successfully")

        # Not needed in test_minimum but we check if it doesn't fails
        try:
            force_model.close_written_info_data_file()
        except AttributeError:
            logger.info(
                "I can't close the data file for the written_info in the visualization."
                " Maybe there is nothing to close so maybe you don't want this data?"
            )

        files = os.listdir(self.outfolder)
        if "written_info_data.txt" in files:
            with open(self.outfolder + "/written_info_data.txt", "r") as f:
                data = []
                for line in f:
                    data.append(line.strip())
            written_info_data = data[:: run_params["write_interval_fac"]]
            os.remove(self.outfolder + "/written_info_data.txt")
            with open(self.outfolder + "/written_info_data.pick", "wb") as f:
                pickle.dump(written_info_data, f)

        try:
            force_model.close_vision_cone_data_file()
        except AttributeError:
            logger.info(
                "I can't close the data file for the vision_cones in the visualization."
                " Maybe there is nothing to close so maybe you don't want this data?"
            )

        files = os.listdir(self.outfolder)
        if "vision_cone_data.pick" in files:
            with open(self.outfolder + "/vision_cone_data.pick", "rb") as f:
                data = []
                while True:
                    try:
                        thing_data = pickle.load(f)
                        data.append(thing_data)
                    except EOFError:
                        break
            vision_cone_data = data[:: run_params["write_interval_fac"]]
            os.remove(self.outfolder + "/vision_cone_data.pick")
            with open(self.outfolder + "/vision_cone_data.pick", "wb") as f:
                pickle.dump(vision_cone_data, f)

        # ureg can sadly not be saved by pickle
        md_params_without_ureg = copy.deepcopy(md_params)
        md_params_without_ureg.ureg = None
        params_to_write = {
            "type": type(force_model),
            "md_params": md_params_without_ureg,
            "col_params": col_params,
            "run_params": run_params,
            "rod_params": rod_params,
            "maze_params": maze_params,
        }

        utils.write_params(
            self.outfolder,
            self.simulation_name,
            params_to_write,
            write_espresso_version=True,
        )
        logger.info(
            "Writing parameters and additional data"
            " from the simulation to the files finished."
        )


class TestFullSimVisualization(ut.TestCase):
    """
    Functional test, also to be used as an example for simulation scripts
    """

    # Parameters usually provided as commandline input
    loglevel_terminal = "info"
    seed = 42

    def test_minimum(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.simulation_name = "test_minimum"
            self.outfolder = utils.setup_sim_folder(temp_dir, self.simulation_name)
            self.mode = "minimum"
            # each simulation needs its own process
            sim = Simulation(
                self.outfolder,
                self.simulation_name,
                self.loglevel_terminal,
                self.seed,
                self.mode,
            )
            process = Process(target=sim.simulate_model)
            process.start()
            time.sleep(30)
            process.terminate()
            if process.exception:
                error, traceback = process.exception
                print(traceback)
            self.assertEqual(process.exception, None)

            # visualization #
            param_fname = f"{self.outfolder}/params_{self.simulation_name}.pick"
            with open(param_fname, "rb") as param_file:
                parameters = pickle.load(param_file)
            ureg = pint.UnitRegistry()

            files = os.listdir(self.outfolder)
            if "trajectory.hdf5" in files:
                positions, directors, times, ids, types, ureg = load_traj_vis(
                    self.outfolder, ureg
                )

            fig, ax = plt.subplots(figsize=(7, 7))
            # setup the units for automatic ax_labeling

            positions.ito(ureg.micrometer)
            times.ito(ureg.second)

            ani_instance = Animations(
                fig,
                ax,
                positions,
                directors,
                times,
                ids,
                types,
                vision_cone_boolean=[False, False, False],
                cone_radius=5,
                n_cones=5,
                cone_half_angle=np.pi / 2,
                cone_vision_of_types=[parameters["col_params"]["col_particle_type"]],
                trace_boolean=[True, True, True],
                trace_fade_boolean=[True, True, True],
                eyes_boolean=[False, False, False],
                arrow_boolean=[True, False, False],
                radius_col=[
                    parameters["col_params"]["col_radius"].magnitude,
                    0,
                    0,
                ],
                schmell_boolean=True,
                schmell_ids=[5],
                maze_boolean=False,
            )

            if self.outfolder is not None:
                load_extra_data_to_visualization(ani_instance, self.outfolder)
            else:
                raise Exception(
                    "You need to specify where your extradata for visualization is"
                    " located.It is assumed that it lies where the trajectory.hdf5 file"
                    " is located."
                )

            ani_instance.animation_plt_init()

            ani_instance.ax.grid(True)

            # set start and end of visualization and set the interval of between frames
            begin_frame = 1
            end_frame = len(times[:])
            ani = FuncAnimation(
                fig,
                ani_instance.animation_plt_update,
                frames=range(begin_frame, end_frame),
                blit=True,
                interval=10,
            )
            # plt.show()
            ani.save("animation.mp4", fps=60)
            # dummy assert
            self.assertEqual("dummy assert", "dummy assert")

    def test_multiple_types(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.simulation_name = "test_multiple_types"
            self.outfolder = utils.setup_sim_folder(temp_dir, self.simulation_name)
            self.mode = "multiple_types"
            # each simulation needs its own process
            sim = Simulation(
                self.outfolder,
                self.simulation_name,
                self.loglevel_terminal,
                self.seed,
                self.mode,
            )
            process = Process(target=sim.simulate_model)
            process.start()
            time.sleep(360)
            process.terminate()
            if process.exception:
                error, traceback = process.exception
                print(traceback)
            self.assertEqual(process.exception, None)

            # visualization #
            param_fname = f"{self.outfolder}/params_{self.simulation_name}.pick"
            with open(param_fname, "rb") as param_file:
                parameters = pickle.load(param_file)

            ureg = pint.UnitRegistry()

            files = os.listdir(self.outfolder)
            if "trajectory.hdf5" in files:
                positions, directors, times, ids, types, ureg = load_traj_vis(
                    self.outfolder, ureg
                )

            fig, ax = plt.subplots(figsize=(7, 7))
            # setup the units for automatic ax_labeling

            positions.ito(ureg.micrometer)
            times.ito(ureg.second)

            ani_instance = Animations(
                fig,
                ax,
                positions,
                directors,
                times,
                ids,
                types,
                vision_cone_boolean=[True, False, False],
                cone_radius=10,
                n_cones=5,
                cone_half_angle=np.pi / 2,
                cone_vision_of_types=[parameters["col_params"]["col_particle_type"], 1],
                trace_boolean=[True, True, True],
                trace_fade_boolean=[False, True, True],
                eyes_boolean=[True, False, False],
                arrow_boolean=[True, True, False],
                radius_col=[
                    parameters["col_params"]["col_radius"].magnitude,
                    parameters["col_params"]["col_radius"].magnitude,
                    0,
                ],
                schmell_boolean=False,
                schmell_ids=[5],
                maze_boolean=False,
            )

            if self.outfolder is not None:
                load_extra_data_to_visualization(ani_instance, self.outfolder)
            else:
                raise Exception(
                    "You need to specify where your extradata for "
                    " visualization is located. It is assumed that it lies where the "
                    " trajectory.hdf5 file is located."
                )

            ani_instance.animation_plt_init()

            ani_instance.ax.grid(True)

            # set start and end of visualization and set the interval of between frames
            begin_frame = 1
            end_frame = len(times[:])
            ani = FuncAnimation(
                fig,
                ani_instance.animation_plt_update,
                frames=range(begin_frame, end_frame),
                blit=True,
                interval=10,
            )
            # plt.show()
            ani.save("animation.mp4", fps=60)
            # dummy assert
            self.assertEqual("dummy assert", "dummy assert")

    def test_rod(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.simulation_name = "test_rod"
            self.outfolder = utils.setup_sim_folder(temp_dir, self.simulation_name)
            self.mode = "rod"
            # each simulation needs its own process
            sim = Simulation(
                self.outfolder,
                self.simulation_name,
                self.loglevel_terminal,
                self.seed,
                self.mode,
            )
            process = Process(target=sim.simulate_model)
            process.start()
            time.sleep(450)
            process.terminate()
            if process.exception:
                error, traceback = process.exception
                print(traceback)
            self.assertEqual(process.exception, None)

            # visualization #
            param_fname = f"{self.outfolder}/params_{self.simulation_name}.pick"
            with open(param_fname, "rb") as param_file:
                parameters = pickle.load(param_file)
            ureg = pint.UnitRegistry()

            files = os.listdir(self.outfolder)
            if "trajectory.hdf5" in files:
                positions, directors, times, ids, types, ureg = load_traj_vis(
                    self.outfolder, ureg
                )

            fig, ax = plt.subplots(figsize=(7, 7))
            # setup the units for automatic ax_labeling

            positions.ito(ureg.micrometer)
            times.ito(ureg.second)

            ani_instance = Animations(
                fig,
                ax,
                positions,
                directors,
                times,
                ids,
                types,
                vision_cone_boolean=[True, False, False],
                cone_radius=10,
                n_cones=5,
                cone_half_angle=np.pi / 2,
                cone_vision_of_types=[
                    parameters["col_params"]["col_particle_type"],
                    parameters["rod_params"]["rod_particle_type"],
                ],
                trace_boolean=[True, True, True],
                trace_fade_boolean=[True, True, True],
                eyes_boolean=[True, False, False],
                arrow_boolean=[False, False, False],
                radius_col=[
                    parameters["col_params"]["col_radius"].magnitude,
                    parameters["rod_params"]["rod_thickness"].magnitude / 2,
                    0,
                ],
                schmell_boolean=False,
                schmell_ids=[5],
                maze_boolean=False,
            )

            if self.outfolder is not None:
                load_extra_data_to_visualization(ani_instance, self.outfolder)
            else:
                raise Exception(
                    "You need to specify where your extradata for "
                    " visualization is located. It is assumed that it lies where the "
                    " trajectory.hdf5 file is located."
                )

            ani_instance.animation_plt_init()

            ani_instance.ax.grid(True)

            # set start and end of visualization and set the interval of between frames
            begin_frame = 1
            end_frame = len(times[:])
            ani = FuncAnimation(
                fig,
                ani_instance.animation_plt_update,
                frames=range(begin_frame, end_frame),
                blit=True,
                interval=10,
            )
            # plt.show()
            ani.save("animation.mp4", fps=60)
            # dummy assert
            self.assertEqual("dummy assert", "dummy assert")

    def test_maze(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.simulation_name = "test_maze"
            self.outfolder = utils.setup_sim_folder(temp_dir, self.simulation_name)
            self.mode = "maze"
            # each simulation needs its own process
            sim = Simulation(
                self.outfolder,
                self.simulation_name,
                self.loglevel_terminal,
                self.seed,
                self.mode,
            )
            process = Process(target=sim.simulate_model)
            process.start()
            time.sleep(30)
            process.terminate()
            if process.exception:
                error, traceback = process.exception
                print(traceback)
            self.assertEqual(process.exception, None)

            # visualization #
            param_fname = f"{self.outfolder}/params_{self.simulation_name}.pick"
            with open(param_fname, "rb") as param_file:
                parameters = pickle.load(param_file)
            ureg = pint.UnitRegistry()

            files = os.listdir(self.outfolder)
            if "trajectory.hdf5" in files:
                positions, directors, times, ids, types, ureg = load_traj_vis(
                    self.outfolder, ureg
                )

            fig, ax = plt.subplots(figsize=(7, 7))
            # setup the units for automatic ax_labeling

            positions.ito(ureg.micrometer)
            times.ito(ureg.second)

            ani_instance = Animations(
                fig,
                ax,
                positions,
                directors,
                times,
                ids,
                types,
                vision_cone_boolean=[True, False, False],
                cone_radius=5,
                n_cones=5,
                cone_half_angle=np.pi / 2,
                cone_vision_of_types=[parameters["col_params"]["col_particle_type"]],
                trace_boolean=[True, True, True],
                trace_fade_boolean=[True, True, True],
                eyes_boolean=[False, False, False],
                arrow_boolean=[True, False, False],
                radius_col=[
                    parameters["col_params"]["col_radius"].magnitude,
                    0,
                    0,
                ],
                schmell_boolean=False,
                schmell_ids=[5],
                maze_boolean=True,
            )

            if self.outfolder is not None:
                load_extra_data_to_visualization(ani_instance, self.outfolder)
            else:
                raise Exception(
                    "You need to specify where your extradata for "
                    " visualization is located. It is assumed that it lies where the "
                    " trajectory.hdf5 file is located."
                )

            ani_instance.animation_plt_init()
            maze_folder = None
            maze_file_name = None
            ani_instance.animation_maze_setup(
                maze_folder,
                maze_file_name,
                parameters["maze_params"]["maze_dic"],
                parameters["maze_params"]["maze_walls"],
            )
            ani_instance.ax.plot(
                parameters["maze_params"]["maze_dic"]["destination"][0],
                parameters["maze_params"]["maze_dic"]["destination"][1],
                "xr",
            )
            ani_instance.ax.grid(True)
            # set start and end of visualization and set the interval of between frames
            begin_frame = 1
            end_frame = len(times[:])
            ani = FuncAnimation(
                fig,
                ani_instance.animation_plt_update,
                frames=range(begin_frame, end_frame),
                blit=True,
                interval=10,
            )
            # plt.show()
            ani.save("animation.mp4", fps=60)
            # dummy assert
            self.assertEqual("dummy assert", "dummy assert")


if __name__ == "__main__":
    ut.main()
