import util
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def moving_average(x, width=10):
    return np.convolve(x, np.ones(width) / width, 'valid')


def main(args):
    if not os.path.exists(args.diagnostics_dir):
        os.makedirs(args.diagnostics_dir)

    fig, axs = plt.subplots(1, 15, figsize=(15 * 3, 3), dpi=100)
    for algorithm in ['mws', 'rws']:
        checkpoint_path = '{}_{}.pt'.format(
            args.checkpoint_path_prefix, algorithm)
        (_, _, theta_losses, phi_losses, cluster_cov_distances,
         test_log_ps, test_log_ps_true, test_kl_qps, test_kl_pqs, test_kl_qps_true, test_kl_pqs_true,
         train_log_ps, train_log_ps_true, train_kl_qps, train_kl_pqs, train_kl_qps_true,
         train_kl_pqs_true) = util.load_checkpoint(
            checkpoint_path, torch.device('cpu'))

        ax = axs[0]
        ax.plot(moving_average(theta_losses), label=algorithm)
        ax.set_xlabel('iteration')
        ax.set_ylabel('theta loss')
        ax.set_xticks([0, len(theta_losses) - 1])

        ax = axs[1]
        ax.plot(moving_average(phi_losses), label=algorithm)
        ax.set_xlabel('iteration')
        ax.set_ylabel('phi loss')
        ax.set_xticks([0, len(phi_losses) - 1])

        ax = axs[2]
        ax.plot(cluster_cov_distances, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('cluster cov distance')
        ax.set_xticks([0, len(cluster_cov_distances) - 1])

        ax = axs[3]
        ax.plot(test_log_ps, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('test log p')
        ax.set_xticks([0, len(test_log_ps) - 1])

        ax = axs[4]
        ax.plot(test_log_ps_true, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('test log p true')
        ax.set_xticks([0, len(test_log_ps_true) - 1])

        ax = axs[5]
        ax.plot(test_kl_qps, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('test kl(q, p)')
        ax.set_xticks([0, len(test_kl_qps) - 1])

        ax = axs[6]
        ax.plot(test_kl_pqs, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('test kl(p, q)')
        ax.set_xticks([0, len(test_kl_pqs) - 1])

        ax = axs[7]
        ax.plot(test_kl_qps_true, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('test kl(q, p true)')
        ax.set_xticks([0, len(test_kl_qps_true) - 1])

        ax = axs[8]
        ax.plot(test_kl_pqs_true, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('test kl(p true, q)')
        ax.set_xticks([0, len(test_kl_pqs_true) - 1])

        ax = axs[9]
        ax.plot(train_log_ps, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('train log p')
        ax.set_xticks([0, len(train_log_ps) - 1])

        ax = axs[10]
        ax.plot(train_log_ps_true, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('train log p true')
        ax.set_xticks([0, len(train_log_ps_true) - 1])

        ax = axs[11]
        ax.plot(train_kl_qps, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('train kl(q, p)')
        ax.set_xticks([0, len(train_kl_qps) - 1])

        ax = axs[12]
        ax.plot(train_kl_pqs, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('train kl(p, q)')
        ax.set_xticks([0, len(train_kl_pqs) - 1])

        ax = axs[13]
        ax.plot(train_kl_qps_true, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('train kl(q, p true)')
        ax.set_xticks([0, len(train_kl_qps_true) - 1])

        ax = axs[14]
        ax.plot(train_kl_pqs_true, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('train kl(p true, q)')
        ax.set_xticks([0, len(train_kl_pqs_true) - 1])

    axs[-1].legend()
    for ax in axs:
        sns.despine(ax=ax, trim=True)

    fig.tight_layout(pad=0)
    path = os.path.join(args.diagnostics_dir, 'losses.pdf')
    fig.savefig(path, bbox_inches='tight')
    print('Saved to {}'.format(path))
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint-path-prefix', default='checkpoint',
                        help=' ')
    parser.add_argument('--diagnostics-dir', default='diagnostics/', help=' ')
    args = parser.parse_args()
    main(args)
