#!/bin/bash
#SBATCH --job-name=cnn_train
# #SBATCH --partition=p.gpu
# #SBATCH --gres=gpu:2
# #SBATCH --partition=p.gpu.ampere
# #SBATCH --gres=gpu:a100:4
#SBATCH --partition=p.test
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
# #SBATCH --time=03:00:00
#SBATCH --output=job.log
#SBATCH --error=job.err

echo "Starting CNN training..."
python3 -u pdf_cnn.py
# python3 -u all_flux_cnn.py

echo "Job complete."
