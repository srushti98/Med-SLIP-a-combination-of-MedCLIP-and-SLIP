#!/bin/bash
#
#SBATCH --job-name=training
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes 1
#SBATCH --account=csci_ga_3033_083-2023fa
#SBATCH --partition=n1s8-v100-1
##SBATCH --ntasks-per-node=2
##SBATCH --cpus-per-task=5
#SBATCH --time=20:10:00
##SBATCH --mem=64GB
#SBATCH --mail-type=END
#SBATCH --output=%jmain.out
#SBATCH --error=%jmaintester.err

##module purge
## Please change python file name in line 18
## change the gpu to rtx1000 later

singularity exec --nv --bind /scratch/sxp8182  --overlay /scratch/sxp8182/MLHC/ML_env/overlay-50G-10M.ext3:ro /share/apps/images/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash \
 -c "source /ext3/env.sh; cd /scratch/sxp8182/MLHC/ML_env/MLHC/; python linear_probe_CBIS.py --path /scratch/sxp8182/MLHC/ML_env/MLHC/linear_logs_mass/img_enc_change_medclip_all_roco_new_mass --ipath /scratch/sxp8182/MLHC/ML_env/MLHC/img_enc_change_medclip_all_roco_new/image_encoder.pth  --epochs 50"
