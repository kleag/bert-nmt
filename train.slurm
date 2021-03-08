#!/bin/bash

# Nombre de machine ou NODES typiquement=1 sauf
#SBATCH -N 1

# Nombre de processus en general=1 (a mémoire distribues type miprun)
#SBATCH --ntasks=1

#SBATCH --gres=gpu:4

# Nom de votre job afficher dans la lise par squeue
#SBATCH --job-name=bert_nmt_train

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

export NCCL_DEBUG=INFO
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

src=de
tgt=en
bedropout=0.5
ARCH=transformer_s2_iwslt_de_en
DATAPATH=iwslt_de_en
SAVEDIR=checkpoints/iwed_${src}_${tgt}_${bedropout}
mkdir -p $SAVEDIR
if [ ! -f "$SAVEDIR/checkpoint_last.pt" ]
then
warmup="--warmup-from-nmt --reset-lr-scheduler"
else
warmup=""
fi
warmup=""
python train.py $DATAPATH \
-a $ARCH --optimizer adam --lr 0.0005 -s $src -t $tgt --label-smoothing 0.1 \
--dropout 0.3 --max-tokens 4000 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --max-update 150000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
--adam-betas '(0.9,0.98)' --save-dir $SAVEDIR --share-all-embeddings $warmup \
--encoder-bert-dropout --encoder-bert-dropout-ratio $bedropout \
--bert-model-name bert-base-multilingual-cased 

wait

echo "Slurm script Done."

