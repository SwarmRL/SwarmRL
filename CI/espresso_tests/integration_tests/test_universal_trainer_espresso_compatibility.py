import numpy as np
import optax

import swarmrl as srl
from CI.espresso_tests.integration_tests.test_rl_trainers import (
    ActoCriticNet,
    EspressoTestRLTrainers,
    actions,
    scale_function,
)


class EspressoTestUniversalRLTrainers(EspressoTestRLTrainers):
    """Run the same trainer CI scenarios against UniversalTrainer."""

    continuous_trainer_cls = srl.trainers.UniversalTrainer
    episodic_trainer_cls = srl.trainers.UniversalTrainer

    def setUp(self):
        self.system = EspressoTestRLTrainers.system
        self.actor_critic = ActoCriticNet()
        self.exploration_policy = srl.exploration_policies.RandomExploration(
            probability=0.0
        )
        self.sampling_strategy = srl.sampling_strategies.GumbelDistribution()
        self.network = srl.networks.FlaxModel(
            flax_model=self.actor_critic,
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1,),
            sampling_strategy=self.sampling_strategy,
            exploration_policy=self.exploration_policy,
        )
        self.task = srl.tasks.searching.GradientSensing(
            source=np.array([500.0, 500.0, 0.0]),
            decay_function=scale_function,
            reward_scale_factor=10,
            box_length=np.array([1000.0, 1000.0, 1000]),
        )
        self.observable = srl.observables.ConcentrationField(
            source=np.array([500.0, 500.0, 0.0]),
            decay_fn=scale_function,
            scale_factor=10000,
            box_length=np.array([1000.0, 1000.0, 1000]),
            particle_type=0,
        )
        self.loss = srl.losses.ProximalPolicyLoss()
        self.agent = srl.agents.ActorCriticAgent(
            particle_type=0,
            network=self.network,
            task=self.task,
            observable=self.observable,
            actions=actions,
            loss=self.loss,
        )
