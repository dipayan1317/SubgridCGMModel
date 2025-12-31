#!/bin/bash
#SBATCH --job-name=cnn_evol_pipeline
#SBATCH --partition=low_unl_1gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=job.log
#SBATCH --error=job.err

echo "Starting CNN training..."
python3 -u flux_cnn.py

echo "Switching to data directory..."
cd ../data/

echo "Starting evolution animation..."
python3 -u mocks/all_evol.py

echo "Job complete."
