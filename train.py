from tokenize import Name
import jax
import optax
import haiku as hk
import jax.numpy as jnp
from typing import NamedTuple
from absl import app
from data import get_cora_dataset
from gat import GAT

# Training hyperparameters
SEED = 42
MAX_STEPS = 10

# Model hyperparameters
NUM_CLASSES = 7


class Batch(NamedTuple):
    nodes_features: jnp.ndarray
    labels: jnp.ndarray
    node_indices: jnp.ndarray


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    rng: jnp.DeviceArray


def main():

    optimiser = optax.adam(3e-4)

    def forward(node_features: jnp.ndarray) -> jnp.ndarray:
        gat = GAT(
            num_layers=2,
            num_heads=[8, 1],
            num_features=[8, NUM_CLASSES],
            dropout=0.6
        )

        return gat(node_features)

    @hk.transform
    def loss_fn(params: hk.Params, batch: Batch) -> jnp.ndarray:
        """Cross-entropy classification loss, regularised by L2 weight decay"""
        logits = forward(batch.node_features)[batch.node_indices]
        targets = jax.nn.one_hot(batch.labels, NUM_CLASSES)[batch.node_indices]

        log_likelihood = jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
        return -log_likelihood

    @jax.jit
    def update(state: TrainingState, batch: Batch):
        loss_and_grad_fn = jax.value_and_grad(loss_fn.apply)
        loss, grads = loss_and_grad_fn(state.params, batch)

        updates, new_opt_state = optimiser.update(grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)

        new_state = TrainingState(new_params, new_opt_state)

        return new_state, loss

    @jax.jit
    def init(rng: jnp.ndarray, batch: Batch) -> TrainingState:
        rng, init_rng = jax.random.split(rng)
        initial_params = loss_fn.init(init_rng, batch.nodes_features)
        initial_opt_state = optimiser.init(initial_params)

        return TrainingState(
            params=initial_params,
            opt_state=initial_opt_state,
            rng=rng
        )

    rng = jax.random.PRNGKey(SEED)

    nodes_features, edge_index, labels, train_indices, val_indices, test_indices = get_cora_dataset()
    train_data = Batch(nodes_features=nodes_features, labels=labels, node_indices=train_indices)
    val_data = Batch(nodes_features=nodes_features, labels=labels, node_indices=val_indices)
    test_data = Batch(nodes_features=nodes_features, labels=labels, node_indices=test_indices)

    training_state = init(rng, train_data)

    # Training loop
    for _ in range(MAX_STEPS):
        state, loss = update(state, train_data)
        print(loss)


if __name__ == '__main__':
    app.run(main)
