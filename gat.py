import jax
import haiku as hk
import jax.numpy as jnp


class GAT(hk.Module):
    def __init__(self, num_layers, num_heads, num_features, dropout=0.5, name=None):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_features = num_features

        self.dropout = dropout

    def __call__(self, node_features, connectivity_mask, is_training=True, return_att_coeffs=False):
        layers = []
        for i in range(self.num_layers):
            layer = GATLayer(
                num_heads=self.num_heads[i],
                num_features=self.num_features[i],
                dropout=self.dropout if is_training else 0,
                return_att_coeffs=return_att_coeffs
            )

            layers.append(layer)

        gat_net = hk.Sequential(layers)

        return gat_net((node_features, connectivity_mask))


class GATLayer(hk.Module):
    def __init__(self, num_heads, num_features, dropout, return_att_coeffs=False, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.num_features = num_features
        self.dropout = dropout
        self.return_att_scores = return_att_coeffs

        self.linear_projection = hk.Linear(self.num_heads * self.num_features)

        # Create initializer for attention weight vectors
        initializer = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")

        # Initialize attention weight vectors
        self.scoring_weight_source = hk.get_parameter(
            "scoring_weight_source",
            (1, self.num_heads, self.num_features),
            init=initializer)
        self.scoring_weight_target = hk.get_parameter(
            "scoring_weight_target",
            (1, self.num_heads, self.num_features),
            init=initializer)

    def __call__(self, x):
        """
        nodes_features: (N, FIN) 
        """

        nodes_features, connectivity_mask, *all_att_coeffs = x

        # Step 1: Linear projection

        # Apply dropout to nodes_features
        nodes_features = hk.dropout(hk.next_rng_key(), self.dropout, nodes_features)

        # Apply linear transformation
        # (N, FIN) * (FIN, NH * FOUT) -> (N, NH, FOUT)
        nodes_features_proj = self.linear_projection(
            nodes_features).reshape(-1, self.num_heads, self.num_features)

        # Apply dropout to projected nodes_features
        nodes_features_proj = hk.dropout(hk.next_rng_key(), self.dropout, nodes_features_proj)

        # Step 2: Calculate attention

        # Shape: (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1)
        scores_source = jnp.sum(nodes_features_proj * self.scoring_weight_source, axis=-1, keepdims=True)
        scores_target = jnp.sum(nodes_features_proj * self.scoring_weight_target, axis=-1, keepdims=True)

        # Shape source: (N, NH, 1) -> (NH, N, 1)
        # Shape target: (N, NH, 1) -> (NH, 1, N)
        scores_source = jnp.transpose(scores_source, (1, 0, 2))
        scores_target = jnp.transpose(scores_target, (1, 2, 0))

        scores = jax.nn.leaky_relu(scores_source + scores_target, negative_slope=0.2)

        attention_coefficients = jax.nn.softmax(scores + connectivity_mask, axis=-1)

        # Step 3: Neighbourhood aggregation

        # Shape: (NH, N, N) * (NH, N, FOUT) -> (NH, N, FOUT)
        nodes_features_out = jnp.matmul(attention_coefficients, jnp.transpose(nodes_features_proj, (1, 0, 2)))

        nodes_features_out = jnp.transpose(nodes_features_out, (1, 0, 2))

        # Step 4: Concatenation

        # Shape: (NH, N, FOUT) -> (N, NH * FOUT)
        nodes_features_out = nodes_features_out.reshape(-1, self.num_heads * self.num_features)

        if self.return_att_scores:
            return nodes_features_out, connectivity_mask, *all_att_coeffs, attention_coefficients
        else:
            return nodes_features_out, connectivity_mask
