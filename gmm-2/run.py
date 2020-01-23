import torch

import util
import train


def run(args):
    # set up args
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        args.cuda = True
    else:
        device = torch.device('cpu')
        args.cuda = False

    util.print_with_time('args = {}'.format(args))

    # init
    true_cluster_cov = torch.eye(args.num_dim, device=device) * 0.03
    generative_model, inference_network, true_generative_model = util.init(
        args.num_data, args.num_clusters, args.num_dim, true_cluster_cov,
        device)

    # data
    data_loader = torch.utils.data.DataLoader(
        true_generative_model.sample_obs(args.num_train),
        batch_size=args.batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(
        true_generative_model.sample_obs(args.num_test),
        batch_size=args.batch_size, shuffle=True)

    # train
    if args.algorithm == 'mws':
        (theta_losses, phi_losses, cluster_cov_distances,
         test_log_ps, test_kls, train_log_ps, train_kls) = train.train_mws(
            generative_model, inference_network, data_loader,
            args.num_iterations, args.memory_size, true_cluster_cov,
            test_data_loader, args.test_num_particles)
    elif args.algorithm == 'rws':
        (theta_losses, phi_losses, cluster_cov_distances,
         test_log_ps, test_kls, train_log_ps, train_kls) = train.train_rws(
            generative_model, inference_network, data_loader,
            args.num_iterations, args.num_particles, true_cluster_cov,
            test_data_loader, args.test_num_particles)

    # save model
    checkpoint_path = '{}_{}.pt'.format(
        args.checkpoint_path_prefix, args.algorithm)
    util.save_checkpoint(checkpoint_path, generative_model, inference_network,
                         theta_losses, phi_losses, cluster_cov_distances,
                         test_log_ps, test_kls, train_log_ps, train_kls)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--algorithm', default='mws', help='rws or mws')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--batch-size', type=int, default=50, help=' ')
    parser.add_argument('--num-dim', type=int, default=2, help=' ')
    parser.add_argument('--num-clusters', type=int, default=3, help=' ')
    parser.add_argument('--num-data', type=int, default=10, help=' ')
    parser.add_argument('--num-train', type=int, default=100, help=' ')
    parser.add_argument('--num-iterations', type=int, default=20000, help=' ')
    parser.add_argument('--num-particles', type=int, default=10, help=' ')
    parser.add_argument('--memory-size', type=int, default=10, help=' ')
    parser.add_argument('--num-test', type=int, default=100, help=' ')
    parser.add_argument('--test-num-particles', type=int, default=5000,
                        help=' ')
    parser.add_argument('--checkpoint-path-prefix', default='checkpoint',
                        help=' ')
    args = parser.parse_args()
    run(args)
