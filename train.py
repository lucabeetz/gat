import jax
import optax
import haiku as hk
import numpy as np
import jax.numpy as jnp
import torch_geometric as tg
from typing import NamedTuple
from absl import app
from data import get_cora_dataset
from gat import GAT

# Training hyperparameters
SEED = 42
MAX_STEPS = 1000

# Model hyperparameters
NUM_CLASSES = 7


class Batch(NamedTuple):
    nodes_features: jnp.ndarray
    labels: jnp.ndarray
    connectivity_mask: jnp.ndarray
    node_indices: np.ndarray


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    rng: jnp.DeviceArray


def gat_fn(batch: Batch) -> jnp.ndarray:
    gat = GAT(
        num_layers=2,
        num_heads=[8, 1],
        num_features=[8, NUM_CLASSES],
        dropout=0.6
    )

    return gat(batch.nodes_features, batch.connectivity_mask)


def main(_):

    gat = hk.transform(gat_fn)
    optimiser = optax.adam(3e-4)

    def loss_fn(params: hk.Params, batch: Batch, rng) -> jnp.ndarray:
        """Cross-entropy classification loss, regularised by L2 weight decay"""
        logits = gat.apply(params, rng, batch)[0].take(batch.node_indices)
        targets = jax.nn.one_hot(batch.labels, NUM_CLASSES).take(batch.node_indices)

        log_likelihood = jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
        return -log_likelihood

    @jax.jit
    def update(state: TrainingState, batch: Batch):
        rng, new_rng = jax.random.split(state.rng)
        loss_and_grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = loss_and_grad_fn(state.params, batch, rng)

        updates, new_opt_state = optimiser.update(grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)

        new_state = TrainingState(new_params, new_opt_state, new_rng)

        return new_state, loss

    @jax.jit
    def init(rng: jnp.ndarray, batch: Batch) -> TrainingState:
        rng, init_rng = jax.random.split(rng)
        initial_params = gat.init(init_rng, batch)
        initial_opt_state = optimiser.init(initial_params)

        return TrainingState(
            params=initial_params,
            opt_state=initial_opt_state,
            rng=rng
        )

    rng = jax.random.PRNGKey(SEED)

    nodes_features, connectivity_mask, labels, train_indices, val_indices, test_indices = get_cora_dataset()

    train_data = Batch(nodes_features=nodes_features, labels=labels,
                       connectivity_mask=connectivity_mask, node_indices=train_indices)
    val_data = Batch(nodes_features=nodes_features, labels=labels,
                     connectivity_mask=connectivity_mask, node_indices=val_indices)
    test_data = Batch(nodes_features=nodes_features, labels=labels,
                      connectivity_mask=connectivity_mask, node_indices=test_indices)

    state = init(rng, train_data)

    # Training loop
    for step in range(MAX_STEPS):
        if step % 100 == 0:
            state, loss = update(state, train_data)

            # Validation
            rng, new_rng = jax.random.split(state.rng)
            logits = gat.apply(state.params, rng, val_data)[0]

            val_acc = jnp.equal(jnp.argmax(logits, axis=-1), val_data.labels).sum() / len(val_data.labels)
            state = TrainingState(state.params, state.opt_state, new_rng)

            print(f'Step: {step}, loss: {loss}, val acc: {val_acc}')


if __name__ == '__main__':
    app.run(main)
