#!/bin/bash
#SBATCH --job-name=ramen_mae_multi       # nom du job
#SBATCH --partition=gpu_p6       # partition GPU
#SBATCH -A qsm@h100
#SBATCH -C h100
##SBATCH --qos=qos_gpu_h100-t4 
#SBATCH --qos=qos_gpu_h100-dev 
#SBATCH --nodes=4                    # on demande un noeud
#SBATCH --ntasks-per-node=4          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:4                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p5)
#SBATCH --cpus-per-task=24           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)

# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=02:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=ramen_mae_multi%j.out      # nom du fichier de sortie
#SBATCH --error=ramen_mae_multi%j.out       # nom du fichier d'erreur (ici commun avec la sortie)
 
# Nettoyage des modules charges en interactif et herites par defaut
module purge

export TMPDIR=$JOBSCRATCH
 
# Chargement des modules
module load arch/h100
module load pytorch-gpu/py3/2.4.0
 
# Echo des commandes lancees
set -x

srun python -u src/train.py exp=SaveCheck trainer=gpu max_epochs=1
