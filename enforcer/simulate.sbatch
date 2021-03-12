#!/bin/bash
#SBATCH --array=0-999
#SBATCH --mail-user=bures@d3s.mff.cuni.cz
#SBATCH --mail-type=END,FAIL
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --partition=volta-lp
ch-run sbt -b /mnt/research/bures -c /home/bures/ftnn-simulator sh runTraceRecord.sh $SLURM_ARRAY_TASK_ID


