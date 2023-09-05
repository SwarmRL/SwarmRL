# #!/usr/bin/env python
# # coding: utf-8
#
# # In[1]:
# import os
#
# import flax.linen as nn
# import jax.numpy as jnp
# import numpy as np
# import optax
# import pint
#
# import swarmrl as srl
# import swarmrl.engine.espresso as espresso
# from swarmrl.gyms import Gym
# from swarmrl.models.interaction_model import Action, Colloid
# from swarmrl.networks.graph_network import GraphModel
# from swarmrl.observables.col_graph import ColGraph, GraphObservable
# from swarmrl.rl_protocols.actor_critic import ActorCritic
# from swarmrl.tasks.searching.species_avoid import SpeciesAvoid
# from swarmrl.networks import FlaxModel
#
#
# def build_circle_cols(n_cols, dist=300):
#     cols = []
#     pos_0 = 1000 * np.random.random(3)
#     pos_0[-1] = 0
#     direction_0 = np.random.random(3)
#     direction_0[-1] = 0
#     for i in range(n_cols - 1):
#         theta = np.random.random(1)[0] * 2 * np.pi
#         position = pos_0 + dist * np.array([np.cos(theta), np.sin(theta), 0])
#         direction = np.random.random(3)
#         direction[-1] = 0
#         direction = direction / np.linalg.norm(direction)
#         cols.append(Colloid(pos=position, director=direction, type=0, id=i))
#     return cols
#
#
# ###################################### General Shit
#
# simulation_name = "sheep shepp"
# seed = np.random.randint(0, 3453276453)
# temperature = 300
#
# n_episodes = 2000
# episode_length = 20
# ppo_epochs = 12
#
#
# ureg = pint.UnitRegistry()
# md_params = espresso.MDParams(
#     ureg=ureg,
#     fluid_dyn_viscosity=ureg.Quantity(8.9e-4, "pascal * second"),
#     WCA_epsilon=ureg.Quantity(273.15, "kelvin") * ureg.boltzmann_constant,
#     temperature=ureg.Quantity(temperature, "kelvin"),
#     box_length=ureg.Quantity(1000, "micrometer"),
#     time_slice=ureg.Quantity(1.0, "second"),  # model timestep
#     time_step=ureg.Quantity(0.01, "second"),  # integrator timestep
#     write_interval=ureg.Quantity(2, "second"),
# )
#
# system_runner = srl.espresso.EspressoMD(
#     md_params=md_params,
#     n_dims=2,
#     seed=seed,
#     out_folder="./sheshep_graph",
#     write_chunk_size=300,
# )
#
# system_runner.add_colloids(
#     3,
#     ureg.Quantity(2.14, "micrometer"),
#     ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
#     ureg.Quantity(400, "micrometer"),
#     type_colloid=1,
# )
#
# system_runner.add_colloids(
#     9,
#     ureg.Quantity(2.14, "micrometer"),
#     ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
#     ureg.Quantity(400, "micrometer"),
#     type_colloid=0,
# )
#
# system_runner.add_sphere(
#     sphere_center=ureg.Quantity(np.array([500, 500, 0]), "micrometer"),
#     radius=ureg.Quantity(500, "micrometer"),
#     sphere_type=2,
# )
#
# translate = Action(force=15.0)
# rotate_clockwise = Action(torque=np.array([0.0, 0.0, 10.0]))
# rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -10.0]))
# do_nothing = Action()
#
# prey_actions = {
#     "Translate": Action(force=15.0),
#     "RotateClockwise": Action(torque=np.array([0.0, 0.0, 10.0])),
#     "RotateCounterClockwise": Action(torque=np.array([0.0, 0.0, -10.0])),
#     "DoNothing": Action(),
# }
#
# preditor_actions = {
#     "Translate": Action(force=10.0),
#     "RotateClockwise": Action(force=5.0, torque=np.array([0.0, 0.0, 10.0])),
#     "RotateCounterClockwise": Action(force=5.0, torque=np.array([0.0, 0.0, -10.0])),
#     "DoNothing": Action(),
# }
#
# # Exploration policy
# exploration_policy = srl.exploration_policies.RandomExploration(probability=0.0)
#
# # Sampling strategy
# sampling_strategy = srl.sampling_strategies.GumbelDistribution()
#
#
# def scale_function(distance: float):
#     """
#     Scaling function for the task
#     """
#     return 1 - distance
#
#
# # Set the task
# prey_task = SpeciesAvoid()
#
#
# ###################################### Graph Shit
#
# cols = build_circle_cols(20)
#
# # Define the observable
# graph_observable = ColGraph(
#     colloids=cols, cutoff=2.0, box_size=np.array([1000, 1000, 1000])
# )
#
# init_graph = graph_observable.compute_initialization_input(cols)
#
# # Define the models.
# graph_actor = GraphModel(init_graph=init_graph, exploration_policy=exploration_policy)
#
# ###################################### RL Shit
#
#
# preditor_task = srl.tasks.searching.SpeciesSearch(
#     decay_fn=scale_function,
#     box_length=np.array([1000.0, 1000.0, 1000]),
#     sensing_type=1,
#     particle_type=0,
# )
#
# preditor_observable = srl.observables.ParticleSensing(
#     decay_fn=scale_function,
#     box_length=np.array([1000.0, 1000.0, 1000]),
#     sensing_type=1,
#     particle_type=0,
# )
#
#
# class GradientActorNet(nn.Module):
#     """A simple dense model."""
#
#     @nn.compact
#     def __call__(self, x):
#         x = nn.Dense(features=128)(x)
#         x = nn.relu(x)
#         y = nn.Dense(features=1)(x)
#         x = nn.Dense(features=4)(x)
#         return x, y
#
#
# gradient_network = FlaxModel(
#     flax_model=GradientActorNet(),
#     optimizer=optax.adam(learning_rate=1e-4),
#     input_shape=(1,),
#     sampling_strategy=sampling_strategy,
#     exploration_policy=exploration_policy,
# )
#
#
# preditor_protocol = srl.rl_protocols.actor_critic.ActorCritic(
#     particle_type=0,
#     network=gradient_network,
#     task=preditor_task,
#     observable=preditor_observable,
#     actions=preditor_actions,
# )
#
# prey_protocol = srl.rl_protocols.ActorCritic(
#     particle_type=1,
#     network=graph_actor,
#     observable=graph_observable,
#     actions=prey_actions,
#     task=prey_task,
# )
#
# # Define the loss model
# ppo_loss = srl.losses.ProximalPolicyLoss(n_epochs=ppo_epochs)
#
#
# rl_trainer = srl.gyms.Gym(
#     [prey_protocol, preditor_protocol],
#     ppo_loss,
# )
#
# # rl_trainer.restore_models('.')
#
# rl_trainer.perform_rl_training(
#     system_runner=system_runner,
#     n_episodes=n_episodes,
#     episode_length=episode_length,
#     episodic_training="episodic 100",
# )
#
# rl_trainer.export_models()
