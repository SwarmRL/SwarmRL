import jax.numpy as jnp

from swarmrl.losses.sac_loss import SoftActorCriticLoss


def test_sac_loss_outputs_finite_scalars():
    loss_fn = SoftActorCriticLoss(target_entropy=-1.0)

    rewards = jnp.ones((8, 1), dtype=jnp.float32)
    dones = jnp.zeros((8, 1), dtype=jnp.float32)
    q1_pred = jnp.full((8, 1), 2.0, dtype=jnp.float32)
    q2_pred = jnp.full((8, 1), 1.5, dtype=jnp.float32)
    q1_next = jnp.full((8, 1), 1.0, dtype=jnp.float32)
    q2_next = jnp.full((8, 1), 0.8, dtype=jnp.float32)
    next_log_probs = jnp.full((8, 1), -0.4, dtype=jnp.float32)
    log_probs = jnp.full((8, 1), -0.3, dtype=jnp.float32)

    losses = loss_fn.compute_losses(
        rewards=rewards,
        dones=dones,
        q1_pred=q1_pred,
        q2_pred=q2_pred,
        q1_next=q1_next,
        q2_next=q2_next,
        next_log_probs=next_log_probs,
        log_probs=log_probs,
        alpha=0.2,
    )

    assert jnp.isfinite(losses["critic_loss"])
    assert jnp.isfinite(losses["actor_loss"])
    assert jnp.isfinite(losses["alpha_loss"])
    assert losses["target_q"].shape == (8, 1)
