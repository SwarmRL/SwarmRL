"""
Run an RL agent to find the center of a box.
"""
import argparse
import copy
import logging

import flax.linen as nn
import h5py as hf
import matplotlib.pyplot as plt
import numpy as np
import optax
import pint

import swarmrl as srl
import swarmrl.utils


def run_analysis():
    """
    Run some analysis.

    Returns
    -------

    """
    with hf.File("example_output/test/trajectory.hdf5") as db:
        data = np.array(db["colloids"]["Unwrapped_Positions"])
    for i in range(len(data[0])):
        plt.plot(data[:, i][:, 0], data[:, i][:, 1])

    plt.show()


def visualize_particles():
    """
    Run a visualization of the particles in the database.

    Returns
    -------

    """
    import znvis as vis

    with hf.File("example_output/test/trajectory.hdf5") as db:
        data = np.array(db["colloids"]["Unwrapped_Positions"])

    mesh = vis.Sphere(radius=10.0, colour=np.array([30, 144, 255]) / 255, resolution=5)
    colloids = vis.Particle(name="Colloid", mesh=mesh, position=data)

    visualizer = vis.Visualizer(particles=[colloids], frame_rate=40)
    visualizer.run_visualization()


def run_simulation():
    """
    Run the simulation.

    Returns
    -------

    """
    # Take user inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("-outfolder_base", default="./example_output")
    parser.add_argument("-name", default="test")
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    outfolder = swarmrl.utils.setup_sim_folder(
        args.outfolder_base, args.name, ask_if_exists=not args.test
    )
    logger = swarmrl.utils.setup_swarmrl_logger(
        f"{outfolder}/{args.name}.log", loglevel_terminal=logging.INFO
    )

    # Define the MD simulation parameters
    ureg = pint.UnitRegistry()
    md_params = srl.espresso.MDParams(
        n_colloids=100,
        ureg=ureg,
        colloid_radius=ureg.Quantity(2.14, "micrometer"),
        fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
        WCA_epsilon=ureg.Quantity(1.0, "kelvin") * ureg.boltzmann_constant,
        colloid_density=ureg.Quantity(2.65, "gram / centimeter**3"),
        temperature=ureg.Quantity(300.0, "kelvin"),
        box_length=ureg.Quantity(1000, "micrometer"),
        initiation_radius=ureg.Quantity(500, "micrometer"),
        time_slice=ureg.Quantity(0.5, "second"),
        time_step=ureg.Quantity(0.5, "second") / 15,
        write_interval=ureg.Quantity(2, "second"),
    )

    model_params = dict(
        target_vel_SI=ureg.Quantity(0.2, "micrometer / second"),
        vision_half_angle=np.pi / 4.0,
    )
    model_params["perception_threshold"] = (
        model_params["vision_half_angle"]
        * md_params.n_colloids
        / (np.pi**2 * md_params.initiation_radius)
    )
    run_params = {"sim_duration": ureg.Quantity(1.5, "hour"), "seed": args.seed}

    md_params_without_ureg = copy.deepcopy(md_params.__dict__)
    md_params_without_ureg.pop("ureg")

    params_to_write = {
        "type": "lavergne",
        "md_params": md_params_without_ureg,
        "model_params": model_params,
        "run_params": run_params,
    }

    swarmrl.utils.write_params(outfolder, args.name, params_to_write)

    # Define the simulation engine.
    system_runner = srl.espresso.EspressoMD(
        md_params=md_params,
        n_dims=2,
        seed=run_params["seed"],
        out_folder=outfolder,
        write_chunk_size=100,
    )
    system_runner.setup_simulation()
    # Define the force model.

    class ActorNet(nn.Module):
        """A simple dense model."""

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=128)(x)
            x = nn.relu(x)
            x = nn.Dense(features=4)(x)
            return x

    class CriticNet(nn.Module):
        """A simple dense model."""

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=128)(x)
            x = nn.relu(x)
            x = nn.Dense(features=1)(x)
            return x

    # Define networks
    critic_stack = CriticNet()
    actor_stack = ActorNet()

    # Define an exploration policy
    exploration_policy = srl.exploration_policies.RandomExploration(probability=0.1)

    # Define a sampling_strategy
    sampling_strategy = srl.sampling_strategies.GumbelDistribution()

    # Value function to use
    value_function = srl.value_functions.ExpectedReturns(gamma=0.99, standardize=True)

    # Define the models.
    actor = srl.networks.FlaxModel(
        flax_model=actor_stack,
        optimizer=optax.adam(learning_rate=0.001),
        input_shape=(1,),
        sampling_strategy=sampling_strategy,
        exploration_policy=exploration_policy,
    )
    critic = srl.networks.FlaxModel(
        flax_model=critic_stack,
        optimizer=optax.adam(learning_rate=0.001),
        input_shape=(1,),
    )

    def scale_function(distance: float):
        """
        Scaling function for the task
        """
        return 1 - distance

    task = srl.tasks.searching.GradientSensing(
        source=np.array([500.0, 500.0, 0.0]),
        decay_function=scale_function,
        reward_scale_factor=10,
        box_size=np.array([1000.0, 1000.0, 1000]),
    )
    observable = task.init_task()

    # Define the loss model
    loss = srl.losses.PolicyGradientLoss(value_function=value_function)

    # Define the force model.
    rl_trainer = srl.gyms.Gym(
        actor,
        critic,
        task,
        loss,
        observable,
        md_params.n_colloids,
    )

    # Run the simulation.
    logger.info("starting simulation")
    n_slices = int(run_params["sim_duration"] / md_params.time_slice)

    n_episodes = 500
    episode_length = int(np.ceil(n_slices / 900))

    rl_trainer.perform_rl_training(
        system_runner=system_runner,
        n_episodes=n_episodes,
        episode_length=episode_length,
        initialize=True,
    )


if __name__ == "__main__":
    """
    Run what you must.
    """
    run_simulation()
    run_analysis()
    # visualize_particles()
