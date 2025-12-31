#!/bin/bash
#SBATCH --job-name=sg_plot
#SBATCH --partition=low_unl_1gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=job.log
#SBATCH --error=job.err

echo "Starting SG Anim..."
python3 -u mocks/mock_sg.py

echo "Done"
