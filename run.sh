#!/bin/bash
#SBATCH --job-name medsam_nuclei_segmentation
#SBATCH --mail-type=BEGIN,END,FAIL            
#SBATCH --mail-user=elahe.ranjbari98@gmail.com
#SBATCH --cpus-per-task 2
#SBATCH --mem 64GB
#SBATCH --output /projects/ovcare/users/elahe_ranjbari/SAM/bmeg591_nuclei_segmentation/output_basesam_random+pos3xneg6n.out
#SBATCH --error /projects/ovcare/users/elahe_ranjbari/SAM/bmeg591_nuclei_segmentation/error_basesam_random+pos3xneg6n.out
#SBATCH -p gpuA6000
#SBATCH --gres=gpu:1
#SBATCH --exclude=dlhost02


cd /projects/ovcare/users/elahe_ranjbari/SAM/bmeg591_nuclei_segmentation
source /projects/ovcare/users/elahe_ranjbari/miniconda3/etc/profile.d/conda.sh
conda activate conda_env

python train_sam.py \
        --data_dir /projects/ovcare/users/elahe_ranjbari/SAM/bmeg591_nuclei_segmentation/NuInsSeg \
        --epochs 50 \
        --batch_size 1 \
        --model_name basesam \
        --prompt_type random