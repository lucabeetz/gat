import numpy as np
import jax.numpy as jnp
import torch_geometric as tg

CORA_TRAIN_RANGE = (0, 140)
CORA_VAL_RANGE = (140, 140+500)
CORA_TEST_RANGE = (1708, 1708+1000)


def get_cora_dataset():
    cora_dataset = tg.datasets.Planetoid(root='data', name='Cora')
    data = cora_dataset[0]

    adj_matrix = jnp.array(tg.utils.to_dense_adj(data.edge_index).squeeze())

    # Add self connections to adj matrix and create connectivity_mask for attention mechanism
    adj_matrix = adj_matrix + jnp.identity(len(adj_matrix))
    connectivity_mask = (adj_matrix == 0) * -jnp.inf

    # Normalise embeddings
    nodes_features = jnp.array(data.x)
    nodes_features_sum_inv = jnp.power(nodes_features.sum(-1), -1)
    nodes_features_normalized = jnp.diag(nodes_features_sum_inv).dot(nodes_features)

    return (
        nodes_features_normalized,
        connectivity_mask,
        jnp.array(data.y),
        np.arange(*CORA_TRAIN_RANGE),
        np.arange(*CORA_VAL_RANGE),
        np.arange(*CORA_TEST_RANGE)
    )
