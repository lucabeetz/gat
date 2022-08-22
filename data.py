from typing import NamedTuple
import numpy as np
import itertools
import jax.numpy as jnp
import torch_geometric as tg
from sklearn.model_selection import train_test_split

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


class KeyValueGraph(NamedTuple):
    edge_list: jnp.ndarray
    nodes_features: jnp.ndarray
    target_mask: jnp.ndarray
    labels: jnp.ndarray


class KeyValueDataset():
    def __init__(self, k: int = 5, train_split: float = 0.8):
        self.k = k
        self.train_split = train_split

    def get_data(self):
        all_data = []

        # Create edge_list for fully connected bipartite graph
        sources = range(0, self.K)
        targets = range(self.K, 2 * self.K)
        edge_list = jnp.array(list(itertools.product(sources, targets)))

        # Create target mask
        target_mask = jnp.array([True] * self.K + [False] * self.K)

        permutations = itertools.permutations(range(self.K))
        for perm in permutations:
            # Create features for all nodes
            top_nodes_features = [(key, self.K) for key in range(self.K)]
            bottom_nodes_features = list(zip(range(self.K), perm))
            nodes_features = jnp.array(top_nodes_features + bottom_nodes_features)

            all_data.append(KeyValueGraph(
                edge_list=edge_list,
                nodes_features=nodes_features,
                target_mask=target_mask,
                labels=jnp.array(perm)
            ))

        return train_test_split(all_data, train_size=self.train_split)
