#!/bin/bash
#
#SBATCH --partition=gpu
#
#SBATCH --job-name=dnaout
#SBATCH --output=dnaout.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

python GettingLastLayer.py
