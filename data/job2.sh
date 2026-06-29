#!/bin/bash
#SBATCH --job-name=mock_sg
# #SBATCH --partition=p.24h
# #SBATCH --partition=p.test	
#SBATCH --partition=p.gpu
#SBATCH --gres=gpu:2
# #SBATCH --partition=p.gpu.ampere
# #SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00
#SBATCH --output=job.log
#SBATCH --error=job.err

FILE_PATH="mocks/sg/gate(16, 8)"

# Remove old mp4s if they exist
rm -f "${FILE_PATH}/all_fields_evolution.mp4"
rm -f "${FILE_PATH}/cons_fields_evolution.mp4"

echo "Starting SG Anim..."
python3 -u mocks/mock_sg.py

echo "Converting gif to mp4..."
module load ffmpeg

ffmpeg -i "${FILE_PATH}/all_fields_evolution.gif" \
       -movflags faststart -pix_fmt yuv420p \
       "${FILE_PATH}/all_fields_evolution.mp4"

ffmpeg -i "${FILE_PATH}/cons_fields_evolution.gif" \
       -movflags faststart -pix_fmt yuv420p \
       "${FILE_PATH}/cons_fields_evolution.mp4"

echo "Deleting GIFs..."
rm -f "${FILE_PATH}/all_fields_evolution.gif"
rm -f "${FILE_PATH}/cons_fields_evolution.gif"

echo "Conversion done."
