#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700

#SBATCH --time=48:00:00

source ~/p1/bin/activate
module purge
module restore p1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python nd_rust/_big_sims_nc1.py