#!/bin/bash
#SBATCH --mail-user=bures@d3s.mff.cuni.cz
#SBATCH --mail-type=END,FAIL
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --partition=volta-hp
ch-run 'python:3.9-buster' -b /mnt/research/bures -c /home/bures/ftnn python3.9 preprocess_json.py 9000 1000 64


