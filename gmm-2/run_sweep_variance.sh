#!/bin/bash
cd gmm-2
for seed in 1 2 3 4 5 6 7 8 9 10
do
  for cluster_variance in 0.03 0.1 0.3 1
  do
	  for num_particles in 2 5 10 20 50
	  do
	    for algorithm in vimco rws
	    do
	      sbatch run.sh $seed $num_particles $algorithm --cluster-variance=$cluster_variance $@
	    done
	  done
	  algorithm=mws

	  num_particles=1
	  memory_size=1
	  sbatch run.sh $seed $num_particles $algorithm --memory-size=$memory_size --cluster-variance=$cluster_variance $@
	  
	  num_particles=3
	  memory_size=2
	  sbatch run.sh $seed $num_particles $algorithm --memory-size=$memory_size --cluster-variance=$cluster_variance $@
	  
	  num_particles=5
	  memory_size=5
	  sbatch run.sh $seed $num_particles $algorithm --memory-size=$memory_size --cluster-variance=$cluster_variance $@
	  
	  num_particles=10
	  memory_size=10
	  sbatch run.sh $seed $num_particles $algorithm --memory-size=$memory_size --cluster-variance=$cluster_variance $@
	  
	  num_particles=25
	  memory_size=25
	  sbatch run.sh $seed $num_particles $algorithm --memory-size=$memory_size $@
    done
done

