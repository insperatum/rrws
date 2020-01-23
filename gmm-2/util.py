import torch
import datetime
import models
import os
from pathlib import Path
from matplotlib import pyplot as plt
# hack for https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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
    true_generative_model = models.GenerativeModel(
        num_data, num_clusters, prior_loc, prior_cov, device,
        true_cluster_cov).to(device)

    return (generative_model, inference_network, true_generative_model)


def save_checkpoint(path, generative_model, inference_network, theta_losses,
                    phi_losses, cluster_cov_distances,
                    test_log_ps, test_kls, train_log_ps, train_kls):
    torch.save({
        'generative_model_state_dict': generative_model.state_dict(),
        'inference_network_state_dict': inference_network.state_dict(),
        'theta_losses': theta_losses,
        'phi_losses': phi_losses,
        'num_data': generative_model.num_data,
        'num_clusters': generative_model.num_clusters,
        'num_dim': generative_model.num_dim,
        'cluster_cov_distances': cluster_cov_distances,
        'test_log_ps': test_log_ps,
        'test_kls': test_kls,
        'train_log_ps': train_log_ps,
        'train_kls': train_kls
    }, path)
    print_with_time('Saved checkpoint to {}'.format(path))


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)

    true_cluster_cov = torch.eye(checkpoint['num_dim'], device=device)
    generative_model, inference_network, _ = init(
        checkpoint['num_data'], checkpoint['num_clusters'],
        checkpoint['num_dim'], true_cluster_cov, device)

    generative_model.load_state_dict(checkpoint['generative_model_state_dict'])
    inference_network.load_state_dict(
        checkpoint['inference_network_state_dict'])
    theta_losses = checkpoint['theta_losses']
    phi_losses = checkpoint['phi_losses']
    cluster_cov_distances = checkpoint['cluster_cov_distances']
    test_log_ps = checkpoint['test_log_ps']
    test_kls = checkpoint['test_kls']
    train_log_ps = checkpoint['train_log_ps']
    train_kls = checkpoint['train_kls']
    return (generative_model, inference_network, theta_losses, phi_losses,
            cluster_cov_distances, test_log_ps, test_kls, train_log_ps,
            train_kls)
