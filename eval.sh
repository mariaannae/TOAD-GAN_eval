#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
##SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=toad
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=mae236@nyu.edu
#SBATCH --output=toad_%j.out

module purge
module load python/intel/3.8.6
module load jdk/11.0.9

cd /scratch/mae236/toad
source env/bin/activate

cd TOAD-CLONE

python3 evaluate.py  --out_ output/wandb/latest-run/files --input-dir input --input-name lvl_1-1.txt --multiproc --not_cuda
