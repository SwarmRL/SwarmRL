"""
Test that ensembled deployment runs.
"""


"""
Class for submitting many jobs in parallel to a cluster.
"""

if __name__ == "__main__":
    import numpy as np

    import swarmrl as srl
    from swarmrl.gyms import SharedNetworkGym
    from swarmrl.networks.graph_network import GraphModel
    from swarmrl.observables.col_graph import ColGraph
    from swarmrl.rl_protocols.shared_ac import SharedActorCritic
    from swarmrl.zum_deploy.run_utils import SimulationHelper

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

    sim_helper = SimulationHelper(colloid_dict=colloid_dict)

    actions = SimulationHelper.get_actions()

    protocol = SharedActorCritic(
        particle_type=0,
        network=graph_actor,
        task=task,
        observable=col_graph,
        actions=actions,
    )

    rl_trainer = SharedNetworkGym(rl_protocols=[protocol])

    system_runner = sim_helper.get_simulation_runner()
    n_episodes = 500
    episode_length = 20

    rl_trainer.perform_rl_training(
        system_runner=system_runner,
        n_episodes=n_episodes,
        episode_length=episode_length,
    )

    rl_trainer.export_models()

    # training_routine = srl.training_routines.EnsembleTraining(
    #     rl_trainer,
    #     get_simulation_runner,
    #     number_of_ensembles=20,
    #     n_episodes=500,
    #     n_parallel_jobs=5,
    #     episode_length=20,
    # )
    # training_routine.train_ensemble()
