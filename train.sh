#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=bayitaas
#SBATCH --mail-type=ALL
#SBATCH --job-name=lineGraph_and_adaptive
#SBATCH --output=lineGraph_and_adaptive_%j.out
#SBATCH --error=lineGraph_and_adaptive_%j.err

cd $SLURM_SUBMIT_DIR

# Use HPC share for Hugging Face cache
export HF_HOME=$HOME/hpc-share/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/hpc-share/.cache/huggingface/transformers
export HF_DATASETS_CACHE=$HOME/hpc-share/.cache/huggingface/datasets
export TMPDIR=$HOME/hpc-share/.tmp
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE $TMPDIR

# Disable tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

# IMPORTANT: Disable Python output buffering
export PYTHONUNBUFFERED=1


echo "Job started at: $(date)"
echo "Line Graph GNN is training..."

# Activate environment
source $HOME/hpc-share/gnn_env/bin/activate

# Run the VIB training
#python Transformer_Fusion_VIB.py
#python Transformer_Fusion_VIB_V2.py
#python VIB_No_Transformer.py
#python MRI_VIB_No_Transformer.py
#python -u ViB_vs_DMIB.py

#python -u Exp1.py

python -u main.py

echo "Job completed at: $(date)"
