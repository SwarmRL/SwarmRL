# %%
import swarmrl.engine.resobee as resobee
import os
infomsg = "I "

import flax.linen as nn
import numpy as np
import optax
import yaml
 
import swarmrl as srl
from swarmrl.actions.actions import Action

# %% [markdown]
# ## RL Configuration

# %%

class ActoCriticNet(nn.Module):
    """A simple dense model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        y = nn.Dense(features=1)(x) #Critic
        x = nn.Dense(features=4)(x) #Actor
        return x, y

# Define an exploration policy
exploration_policy = srl.exploration_policies.RandomExploration(probability=0.1)

# Define a sampling_strategy
sampling_strategy = srl.sampling_strategies.GumbelDistribution()

# Value function to use
value_function = srl.value_functions.ExpectedReturns(
    gamma=0.99, standardize=True
)


#Define the model
actor_critic = ActoCriticNet()
network = srl.networks.FlaxModel(
        flax_model=actor_critic,
        optimizer=optax.adam(learning_rate=0.001),
        input_shape=(1,),
        sampling_strategy=sampling_strategy,
        exploration_policy=exploration_policy,
    )

def scale_function(distance: float):
    """
    Scaling function for the task
    """
    return 1 - distance

task = srl.tasks.searching.GradientSensing(
    source=np.array([10.0, 10.0]),
    decay_function=scale_function,
    reward_scale_factor=100,
    box_length=np.array([20.0, 20.]),
)

observable = srl.observables.ConcentrationField(
    source=np.array([10.0, 10.0]),
    decay_fn=scale_function,
    scale_factor=100,
    box_length=np.array([20.0, 20.0]),
    particle_type=0,
)

# Define the loss model
loss = srl.losses.PolicyGradientLoss(value_function=value_function)

actions = {
    "TranslateLeft": np.array([-10., 0.]),
    "TranslateForward": np.array([0., 10.]),
    "TranslateRight": np.array([10., 0.]),
    "MoveBackwards": np.array([0., -10.]),
}
protocol=srl.agents.ActorCriticAgent(particle_type=0, network=network, task=task, observable=observable,actions=actions,loss=loss
                                     )

# Define the force model.
rl_trainer=srl.trainers.EpisodicTrainer([protocol])


# %%
resobee_root_path = "/tikhome/mgern/Desktop/ResoBee/"

build_path = os.path.join(resobee_root_path, "build")
config_dir = os.path.join(resobee_root_path, 'workflow/projects/debug/parameter-combination-0/seed-0')

target = 'many_body_simulation'
resobee_executable = os.path.join(resobee_root_path, 'build/src', target)


# %%
system_runner = resobee.ResoBee(
    resobee_executable=resobee_executable,
    config_dir=config_dir
)

# %%

with open(os.path.join(config_dir,"config.yaml"), 'r') as file:
    data = yaml.safe_load(file)
episode_length=data["total_time"]/data["integration_time_step"]

rl_trainer.perform_rl_training(
            system_runner=system_runner,
            n_episodes=10,
            episode_length=episode_length,
        )




