import jax
import jax.numpy as np
import numpy.testing as tst

from swarmrl.losses.proximal_policy_loss import ProximalPolicyLoss
from swarmrl.sampling_strategies.gumbel_distribution import GumbelDistribution
from swarmrl.utils.utils import gather_n_dim_indices


class DummyActor:
    def apply_fn(self, params, features):
        out_shape = np.shape(features)
        log_probs = 2.0 * np.ones((out_shape[0], out_shape[1], 4))
        return log_probs

    def __call__(self, params, features):
        out_shape = np.shape(features)
        log_probs = 2.0 * np.ones((out_shape[0], out_shape[1], 4))
        return log_probs


class DummyCritic:
    def __call__(self, features):
        out_shape = np.shape(features)
        values = np.array([np.ones((out_shape[0], out_shape[1]))])
        return values


class TestProximalPolicyLoss:
    """
    Test the loss functions for RL models.
    """

    def test_compute_actor_loss(self):
        """
        Tests if the computed loss is correct.
        It is not tested, if the input dimensions are correct.
        Returns
        -------

        """
        # Create
        epsilon = 0.2
        n_particles = 10
        n_time_steps = 20
        observable_dimension = 4
        value_function = False
        entropy_coefficient = 0.01
        sampling_strategy = GumbelDistribution()

        # Create loss function
        loss = ProximalPolicyLoss(
            value_function=value_function,
            sampling_strategy=sampling_strategy,
            entropy_coefficient=entropy_coefficient,
            record_training=False,
        )

        actor = DummyActor()
        actor_params = 10

        features = np.ones((n_time_steps, n_particles, observable_dimension))

        actions = np.ones((n_time_steps, n_particles), dtype=int)

        # all log_probs are 2 such that the ratio is 1
        old_log_probs_0 = 2 * np.ones((n_time_steps, n_particles))
        # all log_probs are 0 such that the ratio is e^2 > 1 + epsilon
        old_log_probs_1 = 0 * np.ones((n_time_steps, n_particles))
        # all log_probs are 3 such that the ratio is e^-2 < 1 - epsilon
        old_log_probs_2 = 3 * np.ones((n_time_steps, n_particles))

        # all advantages are 1
        advantages = np.ones((n_time_steps, n_particles))
        # all advantages are -1
        advantages2 = -np.ones((n_time_steps, n_particles))

        old_log_probs_list = [old_log_probs_0, old_log_probs_1, old_log_probs_2]
        advantages_list = [advantages, advantages2]

        results = []

        # use the PPO loss function to compute the loss.
        for probs in old_log_probs_list:
            for advantages in advantages_list:
                results.append(
                    loss.compute_actor_loss(
                        actor_params=actor_params,
                        actor=actor,
                        features=features,
                        actions=actions,
                        old_log_probs=probs,
                        advantages=advantages,
                    )
                )

        # now compute the loss by hand.
        def ratio(new_log_props, old_log_probs):
            return np.exp(new_log_props - old_log_probs)

        # the dummy_actor will return just 2
        new_log_probs_all = np.log(
            jax.nn.softmax(actor.apply_fn(actor_params, features))
        )
        new_log_probs = gather_n_dim_indices(new_log_probs_all, actions)
        # calculate the entropy of the new_log_probs
        entropy = np.sum(sampling_strategy.compute_entropy(np.exp(new_log_probs_all)))

        # These results will be compared to the results of the PPO loss function
        true_results = []
        for probs in old_log_probs_list:
            for advantages in advantages_list:
                ratios = ratio(new_log_probs, probs)

                clipped_loss = -1 * np.minimum(
                    ratios * advantages,
                    np.clip(ratios, 1 - epsilon, 1 + epsilon) * advantages,
                )
                loss = np.sum(clipped_loss, 0)
                loss = np.sum(loss) + entropy_coefficient * entropy
                true_results.append(loss)

        # compare the results of the PPO loss function with the results computed by hand
        for i, ppo_loss in enumerate(results):
            tst.assert_almost_equal(ppo_loss, true_results[i], decimal=3)
