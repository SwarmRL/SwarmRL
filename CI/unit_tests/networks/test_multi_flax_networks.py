from tempfile import TemporaryDirectory

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core import unfreeze

from swarmrl.networks.multi_flax_networks import MultiFlaxModel


class TinyNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(4)(x)
        x = nn.relu(x)
        return nn.Dense(2)(x)


def build_model(batch_size=3, seed=0):
    module = TinyNetwork()
    sample_input = jnp.ones((batch_size, 3), dtype=jnp.float32)
    params = unfreeze(module.init(jax.random.PRNGKey(seed), sample_input)["params"])

    model = MultiFlaxModel(seed=seed)
    model.add_network("toy", module, params, optax.adam(1e-3), has_target=True)
    return model, module, params


def test_add_network_clones_target_params_independently():
    model, _, params = build_model(batch_size=2, seed=1)

    original_target_kernel = model.target_params["toy"]["Dense_0"]["kernel"]
    params["Dense_0"]["kernel"] = jnp.zeros_like(params["Dense_0"]["kernel"])

    assert model.states["toy"].step == 0
    assert jnp.array_equal(
        model.target_params["toy"]["Dense_0"]["kernel"], original_target_kernel
    )
    assert not jnp.array_equal(
        model.target_params["toy"]["Dense_0"]["kernel"], params["Dense_0"]["kernel"]
    )


def test_export_and_restore_model_state_round_trip():
    model, module, params = build_model(batch_size=4, seed=2)
    model.target_params["toy"] = jax.tree_util.tree_map(
        lambda x: 2.0 * x, model.target_params["toy"]
    )
    model.states["toy"] = model.states["toy"].apply_gradients(
        grads=jax.tree_util.tree_map(jnp.ones_like, model.states["toy"].params)
    )

    with TemporaryDirectory() as temp_directory:
        model.export_model(filename="multi_flax", directory=temp_directory)

        restored = MultiFlaxModel(seed=7)
        restored_params = unfreeze(
            module.init(jax.random.PRNGKey(11), jnp.ones((4, 3), dtype=jnp.float32))[
                "params"
            ]
        )
        restored.add_network(
            "toy", module, restored_params, optax.adam(1e-3), has_target=True
        )
        restored.restore_model_state(filename="multi_flax", directory=temp_directory)

        assert restored.states["toy"].step == model.states["toy"].step
        assert jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                lambda a, b: jnp.array_equal(a, b),
                restored.states["toy"].params,
                model.states["toy"].params,
            )
        )
        assert jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                lambda a, b: jnp.array_equal(a, b),
                restored.target_params["toy"],
                model.target_params["toy"],
            )
        )
