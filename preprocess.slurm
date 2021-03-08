#!/bin/bash

# Nombre de machine ou NODES typiquement=1 sauf
#SBATCH -N 1

# Nombre de processus en general=1 (a mémoire distribues type miprun)
#SBATCH --ntasks=1

#SBATCH --gres=gpu:1

# Nom de votre job afficher dans la lise par squeue
#SBATCH --job-name=external_knowledge_codegen_train

# Nom du fichier de sortie et des erreurs avec l'id du job
#SBATCH --output=res_%j.log
#####SBATCH --error=res_%j.err

#SBATCH --partition=gpu

# Mail pour etre informe de l'etat de votre job
#SBATCH --mail-type=start,end,fail
#SBATCH --mail-user=gael.de-chalendar@cea.fr

# Temps maximal d'execution du job ci dessous
# d-hh:mm:ss
#SBATCH --time=7-0:00:00

# Taille de la memoire exprime en Mega octets max=190000 ici 50G
#SBATCH --mem=50G

#SBATCH --exclude=node5
####SBATCH --nodelist=node6,node7

#set -o nounset
set -o errexit
set -o pipefail

echo "$0"

# activate environments
source /home/gael/miniconda3/bin/activate
source activate bert-nmt2
# source activate concode

/usr/bin/env python3 --version
if [[ -v SLURM_JOB_ID ]] ; then
  nvidia-smi

  # Affiche la (ou les gpus) allouee par Slurm pour ce job
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi

echo "Begin on machine: `hostname`"

# conda list

EXECUTOR=srun
if [ -z ${SLURM_JOB_ID+x} ]; then
  echo "Not in sbatch"
  EXECUTOR=
fi
echo "EXECUTOR is ${EXECUTOR}"


##############

echo 'Script starting'

cd /home/gael/Projets/Decoder/bert-nmt

# run script
echo 'Pretraining'
${EXECUTOR} python preprocess.py --source-lang en --target-lang de --trainpref wmt17_en_de/train --validpref wmt17_en_de/valid --testpref wmt17_en_de/test   --destdir destdir  --joined-dictionary --bert-model-name bert-base-uncased

wait

echo "Slurm script Done."

