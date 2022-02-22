"""
Run an RL agent to find the center of a box.
"""
import swarmrl as srl
import swarmrl.utils
import logging
import argparse
import copy
import pint
import numpy as np
import tqdm
import torch
import h5py as hf
import matplotlib.pyplot as plt
import znvis as vis


def run_analysis():
    """
    Run some analysis.

    Returns
    -------

    """
    with hf.File('find_center/test/trajectory.hdf5') as db:
        data = np.array(db['colloids']['Unwrapped_Positions'])
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
    with hf.File('find_center/test/trajectory.hdf5') as db:
        data = np.array(db['colloids']['Unwrapped_Positions'])

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
    parser.add_argument('-outfolder_base', default='./find_center')
    parser.add_argument('-name', default='test')
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    outfolder = swarmrl.utils.setup_sim_folder(
        args.outfolder_base, args.name, ask_if_exists=not args.test
    )
    logger = swarmrl.utils.setup_swarmrl_logger(f"{outfolder}/{args.name}.log",
                                                loglevel_terminal=logging.DEBUG)

    # Define the MD simulation parameters
    ureg = pint.UnitRegistry()
    md_params = srl.espresso.MDParams(n_colloids=10,
                                      ureg=ureg,
                                      colloid_radius=ureg.Quantity(2.14, 'micrometer'),
                                      fluid_dyn_viscosity=ureg.Quantity(8.9e-4,
                                                                        'pascal * second'),
                                      WCA_epsilon=ureg.Quantity(293, 'kelvin')
                                                  * ureg.boltzmann_constant,
                                      colloid_density=ureg.Quantity(2.65,
                                                                    'gram / centimeter**3'),
                                      temperature=ureg.Quantity(293, 'kelvin'),
                                      box_length=ureg.Quantity(1000, 'micrometer'),
                                      initiation_radius=ureg.Quantity(106,
                                                                      'micrometer'),
                                      time_slice=ureg.Quantity(0.5, 'second'),
                                      time_step=ureg.Quantity(0.5, 'second') / 15,
                                      write_interval=ureg.Quantity(2, 'second'))

    run_params = {'sim_duration': ureg.Quantity(2, 'hour'),
                  'seed': args.seed}

    md_params_without_ureg = copy.deepcopy(md_params.__dict__)
    md_params_without_ureg.pop('ureg')

    params_to_write = {'md_params': md_params_without_ureg,
                       'run_params': run_params
                       }

    swarmrl.utils.write_params(outfolder, args.name, params_to_write,
                               write_espresso_version=True)

    # Define the simulation engine.
    system_runner = srl.espresso.EspressoMD(md_params=md_params,
                                            n_dims=2,
                                            seed=run_params['seed'],
                                            out_folder=outfolder,
                                            write_chunk_size=1000)
    system_runner.setup_simulation()
    # Define the force model.

    # Define networks
    critic_stack = torch.nn.Sequential(
        torch.nn.Linear(3, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )
    actor_stack = torch.nn.Sequential(
        torch.nn.Linear(3, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 4),
    )

    actor = srl.MLP(actor_stack)
    critic = srl.MLP(critic_stack)
    actor = actor.double()
    critic = critic.double()

    # Set the optimizer.
    critic.optimizer = torch.optim.SGD(critic.parameters(), lr=0.001)
    actor.optimizer = torch.optim.SGD(actor.parameters(), lr=0.001)

    # Define the task
    task = srl.FindOrigin(engine=system_runner, alpha=1.0, beta=0.0, gamma=0.0)

    # Define the loss model
    loss = srl.loss.Loss(md_params.n_colloids)

    observable = srl.PositionObservable()

    # Define the force model.
    force_model = srl.mlp_rl.MLPRL(actor, critic, task, loss, observable)

    # Run the simulation.
    n_slices = int(run_params['sim_duration'] / md_params.time_slice)
    logger.info("Starting simulation")
    for _ in tqdm.tqdm(range(100)):
        system_runner.integrate(int(n_slices / 100), force_model)
        force_model.update_rl()

    system_runner.finalize()


if __name__ == '__main__':
    """
    Run what you must.
    """
    run_simulation()
    run_analysis()
    visualize_particles()
