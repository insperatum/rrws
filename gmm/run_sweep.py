# import subprocess
import argparse
from run import run


def main():
    train_modes = ['ws', 'ww', 'dww', 'vimco', 'reinforce', 'concrete',
                   'relax', 'mws']
    num_particles_list = [2, 5, 10, 20]
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    seeds = [1]

    for seed in seeds:
        for train_mode in train_modes:
            for num_particles in num_particles_list:
                args = argparse.Namespace()
                args.train_mode = train_mode
                args.num_iterations = 100000
                args.logging_interval = 1000
                args.eval_interval = 1000
                args.checkpoint_interval = 1000
                args.batch_size = 100
                args.num_particles = num_particles
                args.num_obss = 100000
                args.mws_memory_size = 10
                args.init_near = False
                args.seed = seed
                args.cuda = False
                run(args)

                # subprocess.call(
                #     'sbatch run.sh {} {} {}'.format(
                #         train_mode, num_particles, seed), shell=True)


if __name__ == '__main__':
    main()
