#!/bin/bash
#SBATCH --mail-user=bures@d3s.mff.cuni.cz
#SBATCH --mail-type=END,FAIL
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --partition=volta-lp
ch-run 'tensorflow.tensorflow:latest-gpu' -b /mnt/research/bures -c /home/bures/ftnn python3 preprocess_otf.py 10000


