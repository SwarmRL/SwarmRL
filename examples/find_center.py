"""
Run an RL agent to find the center of a box.
"""
import swarmrl as srl
import argparse
import copy
from bacteria import utils
import pint
import numpy as np
import tqdm
import torch


n_colloids = 50
# Take user inputs
parser = argparse.ArgumentParser()
parser.add_argument('-outfolder_base', default='./find_center')
parser.add_argument('-name', default='test')
parser.add_argument('-seed', type=int, default=42)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

outfolder, _ = utils.setup_sim_folders(
    args.outfolder_base, args.name, check_existing=not args.test
)

# Define the MD simulation parameters
ureg = pint.UnitRegistry()
md_params = srl.espresso.MDParams(n_colloids=n_colloids,
                                  ureg=ureg,
                                  colloid_radius=ureg.Quantity(2.14, 'micrometer'),
                                  fluid_dyn_viscosity=ureg.Quantity(8.9e-4,
                                                                    'pascal * second'),
                                  WCA_epsilon=ureg.Quantity(293,
                                                            'kelvin') * ureg.boltzmann_constant,
                                  colloid_density=ureg.Quantity(2.65,
                                                                'gram / centimeter**3'),
                                  temperature=ureg.Quantity(293, 'kelvin'),
                                  box_length=ureg.Quantity(1000, 'micrometer'),
                                  initiation_radius=ureg.Quantity(106, 'micrometer'),
                                  time_slice=ureg.Quantity(0.5, 'second'),
                                  time_step=ureg.Quantity(0.5, 'second') / 15,
                                  write_interval=ureg.Quantity(2, 'second'))

model_params = dict(target_vel_SI=ureg.Quantity(0.2, 'micrometer / second'),
                    vision_half_angle=np.pi / 4.
                    )
model_params['perception_threshold'] = model_params[
                                           'vision_half_angle'] * md_params.n_colloids / (
                                               np.pi ** 2 * md_params.initiation_radius)
run_params = {'sim_duration': ureg.Quantity(1.5, 'hour'), 'seed': args.seed}

md_params_without_ureg = copy.deepcopy(md_params.__dict__)
md_params_without_ureg.pop('ureg')

params_to_write = {'type': 'lavergne',
                   'md_params': md_params_without_ureg,
                   'model_params': model_params,
                   'run_params': run_params
                   }

utils.write_params(outfolder, args.name, params_to_write)

# Define the simulation engine.
system_runner = srl.espresso.EspressoMD(md_params=md_params,
                                        n_dims=2,
                                        seed=run_params['seed'],
                                        out_folder=outfolder,
                                        write_chunk_size=1000)
system_runner.setup_simulation()
gamma = system_runner.colloid_friction_translation
target_vel = model_params['target_vel_SI'].m_as('sim_velocity')
act_force = target_vel * gamma

perception_threshold = model_params['perception_threshold'].m_as('1/ sim_length')

# Define the force model.

# Define networks
critic_stack = torch.nn.Sequential(
    torch.nn.Linear(3, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 1),
    torch.nn.ReLU()
)
actor_stack = torch.nn.Sequential(
    torch.nn.Linear(3, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 4),
    torch.nn.ReLU()
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
actions = {
    "Translate": srl.actions.Translate,
    "RotateClockwise": srl.actions.RotateClockwise,
    "RotateCounterClockwise": srl.actions.RotateCounterClockwise,
    "Nothing": srl.actions.DoNothing
}

# Define the loss model
loss = srl.loss.Loss()

observable = srl.PositionObservable()

# Define the force model.
force_model = srl.mlp_rl.MLPRL(actor, critic, task, loss, actions, observable)

# Run the simulation.
n_slices = int(run_params['sim_duration'] / md_params.time_slice)

for _ in tqdm.tqdm(range(1000)):
    system_runner.integrate(int(np.ceil(n_slices / 1000)), force_model)
    force_model.update_rl()

system_runner.finalize()
