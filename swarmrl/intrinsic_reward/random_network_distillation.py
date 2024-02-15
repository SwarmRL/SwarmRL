"""
Reward functions based on Random Network Distillation.

Notes
-----
https://arxiv.org/abs/1810.12894
"""

import jax.numpy as np
from znnl.models.flax_model import FlaxModel
from znnl.models.jax_model import JaxModel

from swarmrl.intrinsic_reward.intrinsic_reward import IntrinsicReward
from swarmrl.intrinsic_reward.rnd_configs import RNDArchitecture, RNDConfig
from swarmrl.utils.colloid_utils import TrajectoryInformation


class RNDReward(IntrinsicReward):
    """
    Class to implement an intrinsic loss based on Random Network Distillation.

    This implementation is based on the neural networks framework implemented in
    the ZnNL library.

    Attributes
    ----------

    """

    def __init__(self, rnd_config: RNDConfig):
        """
        Constructor for the RND Loss class.

        Parameters
        ----------
        See RNDConfig for more information.
        """
        # Automatically unpack the configuration.
        self.__dict__.update(rnd_config.__dict__)

        self.iterations = 0
        self.metric_results = None

        # Initialize the predictor into the training strategy.
        self.target_network: JaxModel = FlaxModel(
            flax_module=RNDArchitecture(),
            optimizer=rnd_config.optimizer,
            input_shape=(1, *rnd_config.input_shape),
        )
        self.predictor_network: JaxModel = FlaxModel(
            flax_module=RNDArchitecture(),
            optimizer=rnd_config.optimizer,
            input_shape=(1, *rnd_config.input_shape),
        )
        self.training_strategy.set_model(self.predictor_network)

    @staticmethod
    def _reshape_data(x: np.ndarray) -> np.ndarray:
        """
        Reshape the data for an equal treatment of time and ensemble.

        Flatten the first two dimensions of the data into a single dimension to treat
        the n_steps and num_colloids equally. This assumes that there is similar
        information available by means of the ensemble of colloids and the time
        evolution of the system.

        Parameters
        ----------
        data : np.ndarray of shape (n_steps, num_colloids, num_features)
                Data to be reshaped.

        Returns
        -------
        reshaped_data : np.ndarray of shape (n_steps * num_colloids, num_features)
                Reshaped data.
        """
        return np.reshape(x, (-1, *np.shape(x)[2:]))

    def compute_distance(self, points: np.ndarray) -> np.ndarray:
        """
        Compute the distance between neural network representations.

        Parameters
        ----------
        points : np.ndarray of shape (1, num_points, num_features)
                Points on which distances should be computed.

        Returns
        -------
        distances : np.ndarray
                A tensor of distances computed using the attached metric.
        """
        x = self._reshape_data(points)
        predictor_predictions = self.predictor_network(x)
        target_predictions = self.target_network(x)

        self.metric_results = self.distance_metric(
            target_predictions, predictor_predictions
        )
        return np.mean(self.metric_results)

    def update(self, episode_data: TrajectoryInformation):
        """
        Udpate RND based on the episode data.

        More specifically, this method will update the predictor network on the
        episode_data.features data.

        Parameters
        ----------
        episode_data : TrajectoryInformation
                A dictionary of episode data.
        """
        domain = self._reshape_data(episode_data.features)
        codomain = self.target_network(domain)
        dataset = {"inputs": domain, "targets": codomain}
        _ = self.training_strategy.train_model(
            train_ds=dataset,
            test_ds=dataset,
            epochs=self.n_epochs,
            batch_size=self.batch_size,
            **self.training_kwargs,
        )

    def compute_reward(self, episode_data: TrajectoryInformation) -> np.ndarray:
        """
        Compute the intrinsic reward of the last state of the episode using RND.

        Parameters
        ----------
        episode_data : TrajectoryInformation
                Information on the trajectory of the agent.

        Returns
        -------
        Reward : np.ndarray of shape (num_colloids, )
                Reward for the current state.
        """
        points = episode_data.features[-1:]
        metric_results = self.compute_distance(points=points)
        if self.clip_rewards is not None:
            metric_results = np.clip(metric_results, *self.clip_rewards)
        return metric_results
