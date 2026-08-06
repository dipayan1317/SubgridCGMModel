#!/bin/bash
#SBATCH --job-name=athena
#SBATCH --output=athena.out
#SBATCH --error=athena.err
# #SBATCH --partition=p.test
#SBATCH --partition=p.gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
# #SBATCH --time=08:00:00

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

export PYTHONPATH=$PWD/python:$PYTHONPATH

# ---- main run ----
# srun ./athena -i kh_cooling_pcunits.athinput -d srct512_256/

# ---- restart run ----
srun ./athena -i kh_cooling_pcunits.athinput -d sct512_256/ -r srct512_256/rst/KH.00005.rst
