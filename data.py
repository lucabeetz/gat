import jax.numpy as jnp
from torch_geometric.datasets import Planetoid


def get_cora_dataset():
    cora_dataset = Planetoid(root='data', name='Cora')
    data = cora_dataset[0]

    return (
        jnp.array(data.x),
        jnp.array(data.edge_index),
        jnp.array(data.y),
        jnp.array(data.train_indices),
        jnp.array(data.val_indices),
        jnp.array(data.test_indices)
    )
