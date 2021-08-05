#!/bin/bash
#SBATCH --job-name=std_rr
##SBATCH --output=out_nmt_reinforce.txt
#SBATCH --mem 4000
#SBATCH --gres=gpu:1
#SBATCH --qos=batch
#SBATCH --ntasks=1
#SBATCH --mail-user=ri197@stud.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=students
##SBATCH --partition=gpushort
#SBATCH --nodelist=gpu09


# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

# JOB STEPS
srun python3.5 run.py --rep bert --mode tune --tune_data_file /home/students/dubey/project/data/NYT/nyt.validation.h5df --test_data_file /home/students/dubey/project/data/NYT/nyt.test.h5df --bert_model_file  /home/students/dubey/project/pacssum_models/pytorch_model_unfinetuned.bin --bert_config_file /home/students/dubey/project/pacssum_models/bert_config.json --bert_vocab_file /home/students/dubey/project/pacssum_models/vocab.txt

