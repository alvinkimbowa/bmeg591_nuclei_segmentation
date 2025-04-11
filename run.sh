#!/bin/bash
#SBATCH --job-name medsam_nuclei_segmentation
#SBATCH --mail-type=BEGIN,END,FAIL            
#SBATCH --mail-user=elahe.ranjbari98@gmail.com
#SBATCH --cpus-per-task 2
#SBATCH --mem 64GB
#SBATCH --output /projects/ovcare/users/elahe_ranjbari/SAM/bmeg591_nuclei_segmentation/output_medsam.out
#SBATCH --error /projects/ovcare/users/elahe_ranjbari/SAM/bmeg591_nuclei_segmentation/error_medsam.out
#SBATCH -p gpuA6000,dgxV100,gpu3090,rtx5000
#SBATCH --gres=gpu:1
#SBATCH --exclude=dlhost02


cd /projects/ovcare/users/elahe_ranjbari/SAM/bmeg591_nuclei_segmentation
source /projects/ovcare/users/elahe_ranjbari/miniconda3/etc/profile.d/conda.sh
conda activate conda_env

python train_sam.py --data_dir /projects/ovcare/users/elahe_ranjbari/SAM/bmeg591_nuclei_segmentation/NuInsSeg --epochs 100 --batch_size 1 --visualize_every 5 --save_every 1