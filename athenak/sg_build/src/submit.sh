#!/bin/bash
#SBATCH --job-name=athena_cpu
#SBATCH --output=athena.out
#SBATCH --error=athena.err
#SBATCH --partition=all_srv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00

# Optional: load any needed modules
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
# ./athena -i sg.athinput -d rh16_8/
./athena -i sg.athinput -d gch_16_8/ -r rh16_8/rst/KH.00005.rst
