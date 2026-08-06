#!/bin/bash
#SBATCH --job-name=cnn_train
#SBATCH --partition=p.gpu
#SBATCH --gres=gpu:2
# #SBATCH --partition=p.gpu.ampere
# #SBATCH --gres=gpu:a100:4
# #SBATCH --partition=p.test
# #SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
# #SBATCH --time=03:00:00
#SBATCH --output=job.log
#SBATCH --error=job.err

echo "Starting CNN training..."
python3 -u hyper_par.py

echo "Job complete."
