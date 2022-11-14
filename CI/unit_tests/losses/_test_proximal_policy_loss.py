import numpy as np
import numpy.testing as tst

from swarmrl.losses.proximal_policy_loss import ProximalPolicyLoss
from swarmrl.sampling_strategies.gumbel_distribution import GumbelDistribution
from swarmrl.utils.utils import gather_n_dim_indices


class DummyActor:
    def apply_fn(self, params, features):
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

        loss = ProximalPolicyLoss(
            value_function=value_function,
            sampling_strategy=sampling_strategy,
            entropy_coefficient=entropy_coefficient,
        )

        actor = DummyActor()
        actor_params = 10
        critic = DummyCritic()

        features = np.ones((n_time_steps, n_particles, observable_dimension))

        actions = np.ones((n_time_steps, n_particles), dtype=int)

        # Get dummy old_log_probs that cover all cases of the ratio.
        # cases are r=1, r>1+epsilon, r<1+epsilon

        # all log_probs are 2 such that the ratio is 1
        # for both positive and negative advantage the loss should be -1 and +1
        old_log_probs_1 = 2 * np.ones((n_time_steps, n_particles))

        # all log_probs are 0 such that the ratio is e^2 > 1 + epsilon
        # for positive advantage the loss should take the clipped version  L = -1.2
        # for negative advantage the loss should take e^2: L = e^2
        old_log_probs_2 = 0 * np.ones((n_time_steps, n_particles))

        # all log_probs are 3 such that the ratio is e^-2 < 1 - epsilon
        # for positive advantage the loss should take the clipped version: L = - e^-2
        # for negative advantage the loss should take e^-2: L = 0.8
        old_log_probs_3 = 3 * np.ones((n_time_steps, n_particles))

        old_log_probs_list = [old_log_probs_1, old_log_probs_2, old_log_probs_3]
        # True values

        # Get some dummy true values such that the A = 1 > 0 and A = -1 < 0
        true_values_1 = 0 * np.ones((n_time_steps, n_particles))  #
        true_values_2 = 2.0 * np.ones((n_time_steps, n_particles))

        true_value_list = [true_values_1, true_values_2]
        results = []
        for probs in old_log_probs_list:
            for values in true_value_list:
                results.append(
                    loss.compute_actor_loss(
                        actor=actor,
                        actor_params=actor_params,
                        critic=critic,
                        features=features,
                        actions=actions,
                        old_log_probs=probs,
                        true_values=values,
                    )
                )

        def ratio(new_log_props, old_log_probs):
            return np.exp(new_log_props - old_log_probs)

        def advantage(pred_val, true_val):
            return true_val - pred_val

        predicted_value = np.squeeze(critic(features=features))

        # the dummy_actor will return just 2
        new_log_probs_all = actor.apply_fn(actor_params, features)
        new_log_probs = gather_n_dim_indices(new_log_probs_all, actions)
        # calculate the entropy of the new_log_probs
        entropy = np.mean(sampling_strategy.compute_entropy(new_log_probs_all))
        # These results were compared to by hand computed values.
        true_results = []
        for probs in old_log_probs_list:
            for values in true_value_list:
                ratios = ratio(new_log_probs, probs)

                advantages = advantage(predicted_value, values)
                # advantages = (advantages - np.mean(advantages)) / (
                #            np.std(advantages) + 1e-10)

                clipped_loss = -1 * np.minimum(
                    ratios * advantages,
                    np.clip(ratios, 1 - epsilon, 1 + epsilon) * advantages,
                )
                loss = np.mean(clipped_loss, 0)
                loss = np.mean(loss) + entropy_coefficient * entropy
                true_results.append(loss)

        for i, loss in enumerate(results):
            tst.assert_almost_equal(loss, true_results[i], decimal=3)
