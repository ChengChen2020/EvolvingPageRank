#!/bin/bash
#
#SBATCH --job-name=pgrank
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=45GB
#
#SBATCH --mail-type=END
#SBATCH --mail-user=cc6858@nyu.edu

cd /scratch/$USER/EvolvingPageRank || exit
python main.py --dataset 'as-caida' --probing_nodes_num 50 --sequence_length 10 --fig_path ./imgs/caida_10_50.png