"""
Run an RL agent to find the center of a box.
"""
import argparse
import copy

import h5py as hf
import matplotlib.pyplot as plt
import numpy as np
import pint
import torch
import znvis as vis
from bacteria import utils
from matplotlib import pyplot as plt

import swarmrl as srl


def run_analysis():
    """
    Run some analysis.

    Returns
    -------

    """
    with hf.File("find_center/test/trajectory.hdf5") as db:
        data = np.array(db["colloids"]["Unwrapped_Positions"])
    time = np.linspace(0, len(data), len(data), dtype=int)
    for i in range(len(data[0])):
        plt.plot(time, np.linalg.norm(data[:, i], axis=1))

    plt.show()


def visualize_particles():
    """
    Run a visualization of the particles in the database.

    Returns
    -------

    """
    with hf.File("find_center/test/trajectory.hdf5") as db:
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
    n_colloids = 10
    # Take user inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("-outfolder_base", default="./find_center")
    parser.add_argument("-name", default="test")
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    outfolder, _ = utils.setup_sim_folders(
        args.outfolder_base, args.name, check_existing=not args.test
    )

    # Define the MD simulation parameters
    ureg = pint.UnitRegistry()
    md_params = srl.espresso.MDParams(
        n_colloids=n_colloids,
        ureg=ureg,
        colloid_radius=ureg.Quantity(2.14, "micrometer"),
        fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
        WCA_epsilon=ureg.Quantity(293, "kelvin") * ureg.boltzmann_constant,
        colloid_density=ureg.Quantity(2.65, "gram / centimeter**3"),
        temperature=ureg.Quantity(293, "kelvin"),
        box_length=ureg.Quantity(1000, "micrometer"),
        initiation_radius=ureg.Quantity(106, "micrometer"),
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
        / (np.pi ** 2 * md_params.initiation_radius)
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

    utils.write_params(outfolder, args.name, params_to_write)

    # Define the simulation engine.
    system_runner = srl.espresso.EspressoMD(
        md_params=md_params,
        n_dims=2,
        seed=run_params["seed"],
        out_folder=outfolder,
        write_chunk_size=1000,
    )
    system_runner.setup_simulation()

    # Define the force model.

    def weights_init_uniform_rule(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)

    # Define networks
    critic_stack = torch.nn.Sequential(
        torch.nn.Linear(3, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )
    actor_stack = torch.nn.Sequential(
        torch.nn.Linear(3, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 4),
        torch.nn.ReLU()
    )

    actor_stack.apply(weights_init_uniform_rule)
    critic_stack.apply(weights_init_uniform_rule)

    actor = srl.networks.MLP(actor_stack)
    critic = srl.networks.MLP(critic_stack)
    actor = actor.double()
    critic = critic.double()

    critic.optimizer = torch.optim.Adam(critic.parameters(), lr=0.01)
    actor.optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)

    # Define the task
    task = srl.tasks.searching.FindLocation(
        side_length=np.array([1000.0, 1000.0, 1000.0]),
        location=np.array([0, 0, 0]),
    )

    # Define the loss model
    loss = srl.losses.ProximalPolicyLoss()

    # Define the observable.
    observable = srl.observables.PositionObservable()

    # Define the force model.
    rl_trainer = srl.models.MLPRL(
        actor, critic, task, loss, observable, n_colloids, outfolder
    )

    # Run the simulation.
    n_slices = int(run_params["sim_duration"] / md_params.time_slice)

    n_episodes = 5000
    episode_length = int(np.ceil(n_slices / 1500))
    actor_weights_list,reward_list = rl_trainer.perform_rl_training(
        system_runner=system_runner,
        n_episodes=n_episodes,
        episode_length=episode_length,
    )
    with open('./simulation_data/actor_weights.txt', 'w') as f:
        print(actor_weights_list, file=f)
    with open('./simulation_data/rewards.txt', 'w') as f:
        print(reward_list, file=f)

    return actor_weights_list,reward_list, n_episodes


if __name__ == "__main__":
    """
    Run what you must.
    """
    actor_weights_list,reward_list, n_episodes = run_simulation()
    print("Actor weights equal: ", np.array_equal(actor_weights_list[0],
                                                  actor_weights_list[-1]))
    x_values = np.linspace(0, n_episodes, n_episodes)
    plt.plot(x_values, reward_list)
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('Rewards')
    plt.show()


    # run_analysis()
    # visualize_particles()
