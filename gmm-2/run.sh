#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH -t 1-00:00:00
#SBATCH --job-name=gmm
#SBATCH -o ./slurm/%j.out
#SBATCH -e ./slurm/%j.err

#cd /om/user/lbh/rrws/gmm-2/

SEED=$1
NUM_PARTICLES=$2
ALGORITHM=$3

BATCH_SIZE=60
NUM_HIDDEN=10

#source /home/tuananh/.bashrc
#which python
#python --version
anaconda-project run python run.py --seed $SEED --num-particles $NUM_PARTICLES --memory-size $NUM_PARTICLES --algorithm $ALGORITHM --num-iterations 50000 --batch-size $BATCH_SIZE --num-hidden $NUM_HIDDEN | tee ./slurm/${SLURM_JOB_ID}_temp.out_err
