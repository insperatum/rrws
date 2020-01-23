import torch
import torch.nn as nn
import losses
import math

from clusterdist import CRP, MaskedSoftmaxClustering


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
    def __init__(self, num_data, prior_loc, prior_cov,
                 device, cluster_cov=None):
        super(GenerativeModel, self).__init__()
        self.num_data = num_data
        self.prior_loc = prior_loc
        self.prior_cov = prior_cov
        self.num_dim = len(self.prior_loc)
        self.pre_cluster_cov = nn.Parameter(
            torch.eye(self.num_dim, device=device))
        self.cluster_cov = cluster_cov
        self.device = device

    def get_cluster_cov(self):
        if self.cluster_cov is None:
            return torch.mm(self.pre_cluster_cov, self.pre_cluster_cov.t())
        else:
            return self.cluster_cov

    def get_latent_dist(self):
        """Returns: distribution with batch shape [] and event shape
            [num_data].
        """
        return CRP(self.num_data)

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

        epsilon = 1e-6 * torch.eye(
            num_data * num_dim, device=self.device)[None]
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

    def get_log_prob(self, latent, obs):
        """Log of joint probability.

        Args:
            latent: tensor of shape [num_particles, batch_size, num_data]
            obs: tensor of shape [batch_size, num_data * num_dim]

        Returns: tensor of shape [num_particles, batch_size]
        """

        num_particles, batch_size, _ = latent.shape
        latent_log_prob = self.get_latent_dist().log_prob(latent)
        obs_dist = self.get_obs_dist(
            latent.view(num_particles * batch_size, -1))
        obs_log_prob = obs_dist.log_prob(
            obs.repeat(num_particles, 1)
        ).reshape(num_particles, batch_size)
        return latent_log_prob + obs_log_prob

    def sample_latent_and_obs(self, num_samples=1):
        """Args:
            num_samples: int

        Returns:
            latent: tensor of shape [num_samples, num_data]
            obs: tensor of shape [num_samples, num_data * num_dim]
        """

        latent_dist = self.get_latent_dist()
        latent = latent_dist.sample((num_samples,))
        obs_dist = self.get_obs_dist(latent)
        obs = obs_dist.sample()

        return latent, obs

    def sample_obs(self, num_samples=1):
        """Args:
            num_samples: int

        Returns:
            obs: tensor of shape [num_samples]
        """

        return self.sample_latent_and_obs(num_samples)[1]


class InferenceNetwork(nn.Module):
    def __init__(self, num_data, num_dim):
        super(InferenceNetwork, self).__init__()
        self.num_data = num_data
        self.num_dim = num_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.num_data * self.num_dim, 16),
            nn.Tanh(),
            nn.Linear(16, self.num_data * self.num_data))

    def get_latent_params(self, obs):
        """Args:
            obs: tensor of shape [batch_size, num_data * num_dim]

        Returns: tensor of shape [batch_size, num_data, num_clusters]
        """
        return self.mlp(obs).reshape(
            -1, self.num_data, self.num_data)

    def get_latent_dist(self, obs):
        """Args:
            obs: tensor of shape [batch_size, num_data * num_dim]

        Returns: distribution with batch shape [batch_size] and event shape
            [num_data]
        """
        logits = self.get_latent_params(obs)
        return MaskedSoftmaxClustering(logits=logits)

    def sample_from_latent_dist(self, latent_dist, num_particles):
        """Samples from q(latent | obs)

        Args:
            latent_dist: distribution with batch shape [batch_size] and event
                shape [num_data]
            num_particles: int

        Returns:
            latent: tensor of shape [num_particles, batch_size, num_data]
        """
        return latent_dist.sample((num_particles,))

    def get_log_prob_from_latent_dist(self, latent_dist, latent):
        """Log q(latent | obs).

        Args:
            latent_dist: distribution with batch shape [batch_size] and event
                shape [num_data]
            latent: tensor of shape [num_particles, batch_size, num_data]

        Returns: tensor of shape [num_particles, batch_size]
        """
        return latent_dist.log_prob(latent)

    def get_log_prob(self, latent, obs):
        """Log q(latent | obs).

        Args:
            latent: tensor of shape [num_particles, batch_size, num_data]
            obs: tensor of shape [batch_size, num_data * num_dim]

        Returns: tensor of shape [num_particles, batch_size]
        """
        return self.get_log_prob_from_latent_dist(
            self.get_latent_dist(obs), latent)


def get_log_p_and_kl(generative_model, inference_network, obs, num_particles):
    """Compute log weight and log prob of inference network.

    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size, num_data * num_dim]
        num_particles: int

    Returns:
        log_p: tensor [batch_size]
        kl: tensor [batch_size]
    """
    log_weight, _ = losses.get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_particles)

    log_p = torch.logsumexp(log_weight, dim=1) - math.log(num_particles)
    elbo = torch.mean(log_weight, dim=1)
    kl = log_p - elbo

    return log_p, kl


def get_log_p_and_kl_enumerate(true_generative_model, generative_model, inference_network, obs):
    """
    Args:
        true_generative_model: models.GenerativeModel object
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size, num_data * num_dim]

    Returns:
        log_p: marginal likelihood tensor [batch_size]
        log_p_true: marginal likelihood tensor [batch_size]
        kl_qp: tensor [batch_size]
        kl_pq: tensor [batch_size]
        kl_qp_true: tensor [batch_size]
        kl_pq_true: tensor [batch_size]
    """
    batch_size = obs.shape[0]
    latent_dist = generative_model.get_latent_dist()
    latent = latent_dist.enumerate()[:, None, :].repeat(1, batch_size, 1)

    log_joint = generative_model.get_log_prob(latent, obs)
    log_p = torch.logsumexp(log_joint, dim=0)
    log_posterior = log_joint - log_p[None, :]

    log_joint_true = true_generative_model.get_log_prob(latent, obs)
    log_p_true = torch.logsumexp(log_joint_true, dim=0)
    log_posterior_true = log_joint_true - log_p_true[None, :]

    log_q = inference_network.get_log_prob(latent, obs)

    kl_qp = torch.sum(torch.exp(log_q) * (log_q - log_posterior), dim=0)
    kl_pq = torch.sum(torch.exp(log_posterior) * (log_posterior - log_q), dim=0)

    kl_qp_true = torch.sum(torch.exp(log_q) * (log_q - log_posterior_true), dim=0)
    kl_pq_true = torch.sum(torch.exp(log_posterior_true) * (log_posterior_true - log_q), dim=0)

    return log_p, log_p_true, kl_qp, kl_pq, kl_qp_true, kl_pq_true


def eval_gen_inf(true_generative_model, generative_model, inference_network, data_loader):
    (log_p_total, log_p_true_total,
     kl_qp_total, kl_pq_total, kl_qp_true_total, kl_pq_true_total) = 0, 0, 0, 0, 0, 0
    num_data = data_loader.dataset.shape[0]
    for obs in iter(data_loader):
        log_p, log_p_true, kl_qp, kl_pq, kl_qp_true, kl_pq_true = get_log_p_and_kl_enumerate(
            true_generative_model, generative_model, inference_network, obs)
        log_p_total += torch.sum(log_p).item() / num_data
        log_p_true_total += torch.sum(log_p_true).item() / num_data
        kl_qp_total += torch.sum(kl_qp).item() / num_data
        kl_pq_total += torch.sum(kl_pq).item() / num_data
        kl_qp_true_total += torch.sum(kl_qp_true).item() / num_data
        kl_pq_true_total += torch.sum(kl_pq_true).item() / num_data
    return (log_p_total, log_p_true_total,
            kl_qp_total, kl_pq_total, kl_qp_true_total, kl_pq_true_total)
