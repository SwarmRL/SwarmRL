import jax.numpy as jnp

from swarmrl.value_functions.td_return_sac import TDReturnsSAC


def test_td_return_sac_uses_done_mask():
    value_fn = TDReturnsSAC(gamma=0.9, standardize=False)
    rewards = jnp.array([[1.0], [2.0]])
    q_next = jnp.array([[10.0], [20.0]])
    logp = jnp.array([[0.0], [0.0]])
    terminated = jnp.array([[0.0], [1.0]])

    target = value_fn(
        rewards, q_next, temperature=0.5, next_log_probs=logp, terminated=terminated
    )

    assert float(target[0, 0]) == 1.0 + 0.9 * 10.0
    assert float(target[1, 0]) == 2.0
