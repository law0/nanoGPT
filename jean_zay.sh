#!/bin/bash
#SBATCH --job-name=tpx_nanoGPT_test # nom du job
#SBATCH --output=tpx_nanoGPT_test%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=tpx_nanoGPT_test%j.err # fichier d’erreur (%j = job ID)
#SBATCH --constraint=h100
#SBATCH --nodes=2 # 2 nodes
#SBATCH --ntasks-per-node=4 # 1 per GPU
#SBATCH --gres=gpu:4 # reserver 4 GPU par noeud
#SBATCH --cpus-per-task=24 # 24 * 4tasks = 96 CPUs (all CPUs of a node)
#SBATCH --time=20:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --qos=qos_gpu_h100-t3

module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut

module load arch/h100 # selectionner les modules compiles pour les H100
module load pytorch-gpu/py3/2.6.0 # charger les modules

set -x # activer l’echo des commandes

srun python3 -u train.py config/train_gpt2.py
