import numpy as np
import pint

import swarmrl as srl
import swarmrl.engine.espresso as espresso
from swarmrl.models.interaction_model import Action


class SimulationHelper:
    def __init__(
        self,
        colloid_dict: dict,
        box_length: float = 1000.0,
        temperature: int = 0,
        add_const_force=None,
        write_params=False,
        add_config_walls=False,
    ):
        self.add_const_force = add_const_force
        self.write_params = write_params
        self.add_config_walls = add_config_walls
        self.colloid_dict = colloid_dict

        self.ureg = pint.UnitRegistry()
        self.md_params = espresso.MDParams(
            ureg=self.ureg,
            fluid_dyn_viscosity=self.ureg.Quantity(8.9e-4, "pascal * second"),
            WCA_epsilon=self.ureg.Quantity(temperature, "kelvin")
            * self.ureg.boltzmann_constant,
            temperature=self.ureg.Quantity(temperature, "kelvin"),
            box_length=self.ureg.Quantity(box_length, "micrometer"),
            time_slice=self.ureg.Quantity(0.5, "second"),  # model timestep
            time_step=self.ureg.Quantity(0.5, "second") / 10,  # integrator timestep
            write_interval=self.ureg.Quantity(2, "second"),
        )

        self.sim_duration = self.ureg.Quantity(3, "minute")
        self.n_slices = int(self.sim_duration / self.md_params.time_slice)

    def _set_episode_length(self, wish_length):
        """
        This function compute the dinominator of the episode length.
        """
        div = np.ceil(self.n_slices / wish_length)
        episode_length = int(np.ceil(self.n_slices / div))
        return episode_length

    def get_actions(self, force=10.0, torque=10.0):
        translate = Action(force=force)
        rotate_clockwise = Action(torque=np.array([0.0, 0.0, torque]))
        rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -1 * torque]))
        do_nothing = Action()

        actions = {
            "RotateClockwise": rotate_clockwise,
            "Translate": translate,
            "RotateCounterClockwise": rotate_counter_clockwise,
            "DoNothing": do_nothing,
        }
        return actions

    def get_simulation_runner(self):
        """
        Collect a simulation runner.
        """
        simulation_name = "training"
        seed = int(np.random.uniform(1, 100))

        system_runner = srl.espresso.EspressoMD(
            md_params=self.md_params,
            n_dims=2,
            seed=seed,
            out_folder=simulation_name,
            write_chunk_size=100,
        )

        for key in self.colloid_dict:
            system_runner.add_colloids(
                n_colloids=self.colloid_dict[key]["number"],
                radius_colloid=self.ureg.Quantity(2.14, "micrometer"),
                random_placement_center=self.ureg.Quantity(
                    self.colloid_dict[key]["init_center"], "micrometer"
                ),
                random_placement_radius=self.ureg.Quantity(
                    self.colloid_dict[key]["init_radius"], "micrometer"
                ),
                type_colloid=key,
            )

        return system_runner
