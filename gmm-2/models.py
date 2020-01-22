import torch
import torch.nn as nn


def batched_kronecker(A, B):
    """
    Args:
        A: tensor [batch_size, a_dim_1, a_dim_2]
        B: tensor [b_dim_1, b_dim_2]

    Returns tensor [batch_size, a_dim_1 * b_dim_1, a_dim_2 * b_dim_2]
    """
    batch_size, a_dim_1, a_dim_2 = A.shape
    b_dim_1, b_dim_2 = B.shape
    return torch.einsum("xab,cd->xacbd", A, B).view(
        batch_size, a_dim_1 * b_dim_1, a_dim_2 * b_dim_2)


class GenerativeModel(nn.Module):
    def __init__(self, num_data, num_clusters, prior_loc, prior_cov):
        super(GenerativeModel, self).__init__()
        self.num_data = num_data
        self.num_clusters = num_clusters
        self.prior_loc = prior_loc
        self.prior_cov = prior_cov
        self.num_dim = len(self.prior_loc)
        self.pre_cluster_cov = nn.Parameter(
            torch.randn((self.num_dim, self.num_dim)))

    def get_cluster_cov(self):
        return torch.mm(self.pre_cluster_cov, self.pre_cluster_cov.t())

    def get_latent_dist(self):
        """Returns: distribution with batch shape [] and event shape
            [num_data].
        """
        return torch.distributions.Independent(
            torch.distributions.Categorical(
                logits=torch.ones(self.num_data, self.num_clusters)),
            reinterpreted_batch_ndims=1)

    def get_obs_dist(self, latent):
        """
        Args:
            latent: tensor [batch_size, num_data]

        Returns: distribution with batch_shape [batch_size] and
            event_shape [num_data * num_dim]
        """

        batch_size, num_data = latent.shape
        cluster_cov = self.get_cluster_cov()
        num_dim = cluster_cov.shape[0]

        loc = self.prior_loc.repeat(self.num_data)

        epsilon = torch.eye(num_data * num_dim)[None] * 1e-6
        cov = (
            batched_kronecker(
                torch.eye(num_data)[None],
                cluster_cov) +
            batched_kronecker(
                (latent[:, :, None] == latent[:, None, :]) * 1,
                self.prior_cov) +
            epsilon
        )

        return torch.distributions.MultivariateNormal(loc, cov)


class InferenceNetwork(nn.Module):
    def __init__(self, num_data, num_clusters, num_dim):
        super(InferenceNetwork, self).__init__()
        self.num_data = num_data
        self.num_clusters = num_clusters
        self.num_dim = num_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.num_data * self.num_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, self.num_data * self.num_clusters))

    def get_latent_params(self, obs):
        """Args:
            obs: tensor of shape [batch_size, num_data * num_dim]

        Returns: tensor of shape [batch_size, num_data, num_clusters]
        """
        return self.mlp(obs).reshape(
            -1, self.num_data, self.num_clusters)

    def get_latent_dist(self, obs):
        """Args:
            obs: tensor of shape [batch_size, num_data * num_dim]

        Returns: distribution with batch shape [batch_size] and event shape
            [num_data]
        """
        logits = self.get_latent_params(obs)
        return torch.distributions.Independent(
            torch.distributions.Categorical(logits=logits),
            reinterpreted_batch_ndims=1)
