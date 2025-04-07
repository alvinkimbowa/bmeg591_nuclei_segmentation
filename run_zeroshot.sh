#!/bin/bash
#SBATCH --job-name sam_nuclei_segmentation
#SBATCH --mail-type=BEGIN,END,FAIL            
#SBATCH --mail-user=elahe.ranjbari98@gmail.com
#SBATCH --cpus-per-task 2
#SBATCH --mem 32GB
#SBATCH --output /projects/ovcare/users/elahe_ranjbari/SAM/bmeg591_nuclei_segmentation/output_zeroshot_sam.out
#SBATCH --error /projects/ovcare/users/elahe_ranjbari/SAM/bmeg591_nuclei_segmentation/error_zeroshot_sam.out
#SBATCH -p gpuA6000,dgxV100,gpu3090,rtx5000
#SBATCH --gres=gpu:1
#SBATCH --exclude=dlhost02


cd /projects/ovcare/users/elahe_ranjbari/SAM/bmeg591_nuclei_segmentation
source /projects/ovcare/users/elahe_ranjbari/miniconda3/etc/profile.d/conda.sh
conda activate conda_env

python zeroshot_sam.py --data_dir /projects/ovcare/users/elahe_ranjbari/SAM/bmeg591_nuclei_segmentation/NuInsSeg --batch_size 1