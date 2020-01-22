import torch
import math
import util


def get_log_weight_and_log_q(generative_model, inference_network, obs,
                             num_particles=1):
    """Compute log weight and log prob of inference network.

    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size, num_data * num_dim]
        num_particles: int

    Returns:
        log_weight: tensor of shape [batch_size, num_particles]
        log_q: tensor of shape [batch_size, num_particles]
    """

    latent_dist = inference_network.get_latent_dist(obs)
    latent = inference_network.sample_from_latent_dist(
        latent_dist, num_particles)
    log_p = generative_model.get_log_prob(latent, obs).transpose(0, 1)
    log_q = inference_network.get_log_prob_from_latent_dist(
        latent_dist, latent).transpose(0, 1)
    log_weight = log_p - log_q
    return log_weight, log_q


def get_wake_theta_loss_from_log_weight(log_weight):
    """Args:
        log_weight: tensor of shape [batch_size, num_particles]

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """

    _, num_particles = log_weight.shape
    elbo = torch.mean(
        torch.logsumexp(log_weight, dim=1) - math.log(num_particles))
    return -elbo, elbo


def get_wake_theta_loss(generative_model, inference_network, obs,
                        num_particles=1):
    """Scalar that we call .backward() on and step the optimizer.

    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size, num_data * num_dim]
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, _ = get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_particles)
    return get_wake_theta_loss_from_log_weight(log_weight)


def get_wake_phi_loss_from_log_weight_and_log_q(log_weight, log_q):
    """Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """
    normalized_weight = util.exponentiate_and_normalize(log_weight, dim=1)
    return torch.mean(-torch.sum(normalized_weight.detach() * log_q, dim=1))


def get_wake_phi_loss(generative_model, inference_network, obs,
                      num_particles=1):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size, num_data * num_dim]
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """
    log_weight, log_q = get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_particles)
    return get_wake_phi_loss_from_log_weight_and_log_q(log_weight, log_q)
