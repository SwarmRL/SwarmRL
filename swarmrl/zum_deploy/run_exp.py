"""
Test that ensembled deployment runs.
"""


"""
Class for submitting many jobs in parallel to a cluster.
"""

if __name__ == "__main__":
    import numpy as np
    import pint

    import swarmrl as srl
    import swarmrl.engine.espresso as espresso
    from swarmrl.gyms import SharedNetworkGym
    from swarmrl.models.interaction_model import Action
    from swarmrl.networks.graph_network import GraphModel
    from swarmrl.observables.col_graph import ColGraph
    from swarmrl.rl_protocols.shared_ac import SharedActorCritic

    def get_simulation_runner():
        """
        Collect a simulation runner.
        """
        box_length = 1000.0
        temperature: int = 0

        ureg = pint.UnitRegistry()
        md_params = espresso.MDParams(
            ureg=ureg,
            fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
            WCA_epsilon=ureg.Quantity(temperature, "kelvin") * ureg.boltzmann_constant,
            temperature=ureg.Quantity(temperature, "kelvin"),
            box_length=ureg.Quantity(box_length, "micrometer"),
            time_slice=ureg.Quantity(0.5, "second"),  # model timestep
            time_step=ureg.Quantity(0.5, "second") / 10,  # integrator timestep
            write_interval=ureg.Quantity(2, "second"),
        )

        simulation_name = "training"
        seed = int(np.random.uniform(1, 100))

        system_runner = srl.espresso.EspressoMD(
            md_params=md_params,
            n_dims=2,
            seed=seed,
            out_folder=simulation_name,
            write_chunk_size=100,
        )

        for key in colloid_dict:
            system_runner.add_colloids(
                n_colloids=colloid_dict[key]["number"],
                radius_colloid=ureg.Quantity(2.14, "micrometer"),
                random_placement_center=ureg.Quantity(
                    colloid_dict[key]["init_center"], "micrometer"
                ),
                random_placement_radius=ureg.Quantity(
                    colloid_dict[key]["init_radius"], "micrometer"
                ),
                type_colloid=key,
            )
        return system_runner

    translate = Action(force=10.0)
    rotate_clockwise = Action(torque=np.array([0.0, 0.0, 10.0]))
    rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -10.0]))
    do_nothing = Action()

    actions = {
        "RotateClockwise": rotate_clockwise,
        "Translate": translate,
        "RotateCounterClockwise": rotate_counter_clockwise,
        "DoNothing": do_nothing,
    }

    # Define the models.
    graph_actor = GraphModel(
        record_memory=False,
    )

    # Define the task
    task = srl.tasks.searching.FromGroup(
        box_length=np.array([1000.0, 1000.0, 1000.0]),
        reward_scale_factor=10000,
    )

    # Define the observable
    col_graph = ColGraph(cutoff=1.2, box_size=np.array([1000, 1000, 1000]))

    colloid_dict = {
        0: {"number": 10, "init_center": np.array([500, 500, 0]), "init_radius": 400}
    }
    # sim_helper = SimulationHelper(colloid_dict=colloid_dict)

    protocol = SharedActorCritic(
        particle_type=0,
        network=graph_actor,
        task=task,
        observable=col_graph,
        actions=actions,
    )

    rl_trainer = SharedNetworkGym(rl_protocols=[protocol])

    # system_runner = sim_helper.get_simulation_runner()
    # n_episodes = 100
    # episode_length = 20
    #
    # rl_trainer.perform_rl_training(
    #     system_runner=system_runner,
    #     n_episodes=n_episodes,
    #     episode_length=episode_length,
    # )

    # rl_trainer.export_models()

    training_routine = srl.training_routines.EnsembleTraining(
        rl_trainer,
        get_simulation_runner,
        number_of_ensembles=20,
        n_episodes=500,
        n_parallel_jobs=5,
        episode_length=20,
    )
    training_routine.train_ensemble()
