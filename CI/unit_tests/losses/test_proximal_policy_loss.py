import jax
import jax.numpy as np
import numpy.testing as tst
import optax

from swarmrl.losses.proximal_policy_loss import ProximalPolicyLoss
from swarmrl.sampling_strategies.gumbel_distribution import GumbelDistribution
from swarmrl.utils.utils import gather_n_dim_indices


class DummyNetwork:
    def apply_fn(self, params, features):
        out_shape = np.shape(features)
        log_probs = 2.0 * np.ones((out_shape[0], out_shape[1], 4))
        values = np.array([np.ones((out_shape[0], out_shape[1]))])
        return log_probs, values

    def __call__(self, params, features):
        return self.apply_fn(params, features)


def dummy_value_function(rewards, values):
    advantages = np.ones_like(rewards)
    sign = np.sign(rewards)

    returns = 2 * np.array([np.ones_like(rewards)])
    return advantages * sign, returns


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
        value_function = dummy_value_function
        entropy_coefficient = 0.01
        sampling_strategy = GumbelDistribution()

        # Create loss function
        loss = ProximalPolicyLoss(
            value_function=value_function,
            sampling_strategy=sampling_strategy,
            entropy_coefficient=entropy_coefficient,
        )

        network = DummyNetwork()
        network_params = 10

        features = np.ones((n_time_steps, n_particles, observable_dimension))

        actions = np.ones((n_time_steps, n_particles), dtype=int)

        # all log_probs are 2 such that the ratio is 1
        old_log_probs_0 = 2 * np.ones((n_time_steps, n_particles))
        # all log_probs are 0 such that the ratio is e^2 > 1 + epsilon
        old_log_probs_1 = 0 * np.ones((n_time_steps, n_particles))
        # all log_probs are 3 such that the ratio is e^-2 < 1 - epsilon
        old_log_probs_2 = 3 * np.ones((n_time_steps, n_particles))

        # all advantages are 1
        rewards = np.ones((n_time_steps, n_particles))
        # all advantages are -1
        rewards2 = np.ones((n_time_steps, n_particles))

        old_log_probs_list = [old_log_probs_0, old_log_probs_1, old_log_probs_2]
        rewards_list = [rewards, rewards2]

        results = []

        # use the PPO loss function to compute the loss.
        for probs in old_log_probs_list:
            for rewards in rewards_list:
                results.append(
                    loss._calculate_loss(
                        network_params=network_params,
                        network=network,
                        feature_data=features,
                        action_indices=actions,
                        old_log_probs=probs,
                        rewards=rewards,
                    )
                )

        # now compute the loss by hand.
        def ratio(new_log_props, old_log_probs):
            return np.exp(new_log_props - old_log_probs)

        new_logits, new_predicted_values = network.apply_fn(network_params, features)
        # the dummy_actor will return just 2
        new_log_probs_all = np.log(jax.nn.softmax(new_logits, axis=-1) + 1e-8)
        new_log_probs = gather_n_dim_indices(new_log_probs_all, actions)
        # calculate the entropy of the new_log_probs
        entropy = np.sum(sampling_strategy.compute_entropy(np.exp(new_log_probs_all)))

        # These results will be compared to the results of the PPO loss function
        true_results = []
        for probs in old_log_probs_list:
            for rewards in rewards_list:
                ratios = ratio(new_log_probs, probs)
                advantages, returns = value_function(rewards, None)
                clipped_loss = -1 * np.minimum(
                    ratios * advantages,
                    np.clip(ratios, 1 - epsilon, 1 + epsilon) * advantages,
                )
                loss = np.sum(clipped_loss, 0)

                critic_loss = optax.huber_loss(new_predicted_values, returns)
                critic_loss = np.sum(critic_loss)
                loss = np.sum(loss) - entropy_coefficient * entropy + 0.5 * critic_loss
                true_results.append(loss)

        # compare the results of the PPO loss function with the results computed by hand
        for i, ppo_loss in enumerate(results):
            tst.assert_almost_equal(ppo_loss, true_results[i], decimal=3)
