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
python main.py --dataset 'as-733' --probing_nodes_num 10 --sequence_length 733 --fig_path ./imgs/as_10.png
#python main.py --dataset 'as-caida' --probing_nodes_num 100 --sequence_length 122 --fig_path ./imgs/caida_100.png