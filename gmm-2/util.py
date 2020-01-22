import torch
import datetime
import models
import os
from pathlib import Path
from matplotlib import pyplot as plt


def lognormexp(values, dim=0):
    """Exponentiates, normalizes and takes log of a tensor.

    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n

    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                                 exp(values[i_1, ..., i_N])
            log( ------------------------------------------------------------ )
                    sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
    # log_numerator = values
    return values - log_denominator


def exponentiate_and_normalize(values, dim=0):
    """Exponentiates and normalizes a tensor.

    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n

    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                            exp(values[i_1, ..., i_N])
            ------------------------------------------------------------
             sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    return torch.exp(lognormexp(values, dim=dim))


def get_yyyymmdd():
    return str(datetime.date.today()).replace('-', '')


def get_hhmmss():
    return datetime.datetime.now().strftime('%H:%M:%S')


def print_with_time(str):
    print(get_yyyymmdd() + ' ' + get_hhmmss() + ' ' + str)


def save_plot(filename, x, label):
    os.makedirs(Path(filename).parent, exist_ok=True)
    plt.clf()
    s = 3
    plt.figure(figsize=(s*len(x), s))
    lim = x.abs().ceil().max()

    for i, (x_i, label_i) in enumerate(zip(x, label)):
        plt.subplot(1, len(x), i+1)
        x_i = x_i.reshape(-1, 2)
        for j in range(label_i.max()+1):
            plt.scatter(x_i[label_i == j, 0], x_i[label_i == j, 1])
            plt.xlim([-lim, lim])
            plt.ylim([-lim, lim])
    plt.savefig(filename)


def init(num_data, num_clusters, num_dim, true_cluster_cov, device):
    prior_loc = torch.zeros(num_dim, device=device)
    prior_cov = torch.eye(num_dim, device=device)
    generative_model = models.GenerativeModel(
        num_data, num_clusters, prior_loc, prior_cov, device).to(device)
    inference_network = models.InferenceNetwork(
        num_data, num_clusters, num_dim).to(device)
    optimizer_theta = torch.optim.Adam(generative_model.parameters())
    optimizer_phi = torch.optim.Adam(inference_network.parameters())

    true_generative_model = models.GenerativeModel(
        num_data, num_clusters, prior_loc, prior_cov, device,
        true_cluster_cov).to(device)

    theta_losses = []
    phi_losses = []
    return (generative_model, inference_network, optimizer_theta,
            optimizer_phi, true_generative_model, theta_losses, phi_losses)
