import util
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns


def main(args):
    if not os.path.exists(args.diagnostics_dir):
        os.makedirs(args.diagnostics_dir)

    fig, axs = plt.subplots(1, 7, figsize=(7 * 3, 3), dpi=100)
    for algorithm in ['mws', 'rws']:
        checkpoint_path = '{}_{}.pt'.format(
            args.checkpoint_path_prefix, algorithm)
        (_, _, theta_losses, phi_losses, cluster_cov_distances,
         test_log_ps, test_kls, train_log_ps,
         train_kls) = util.load_checkpoint(
            checkpoint_path, torch.device('cpu'))

        ax = axs[0]
        ax.plot(theta_losses, label=algorithm)
        ax.set_xlabel('iteration')
        ax.set_ylabel('theta loss')
        ax.set_xticks([0, len(theta_losses) - 1])

        ax = axs[1]
        ax.plot(phi_losses, label=algorithm)
        ax.set_xlabel('iteration')
        ax.set_ylabel('phi loss')
        ax.set_xticks([0, len(phi_losses) - 1])

        ax = axs[2]
        ax.plot(cluster_cov_distances, label=algorithm)
        ax.set_xlabel('iteration')
        ax.set_ylabel('cluster cov distance')
        ax.set_xticks([0, len(cluster_cov_distances) - 1])

        ax = axs[3]
        ax.plot(test_log_ps, label=algorithm)
        ax.set_xlabel('iteration')
        ax.set_ylabel('test log p')
        ax.set_xticks([0, len(test_log_ps) - 1])

        ax = axs[4]
        ax.plot(test_kls, label=algorithm)
        ax.set_xlabel('iteration')
        ax.set_ylabel('test KL')
        ax.set_xticks([0, len(test_kls) - 1])

        ax = axs[5]
        ax.plot(train_log_ps, label=algorithm)
        ax.set_xlabel('iteration')
        ax.set_ylabel('train log p')
        ax.set_xticks([0, len(train_log_ps) - 1])

        ax = axs[6]
        ax.plot(train_kls, label=algorithm)
        ax.set_xlabel('iteration')
        ax.set_ylabel('train KL')
        ax.set_xticks([0, len(train_kls) - 1])

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
