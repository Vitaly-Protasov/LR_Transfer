#!/bin/bash

lang_pairs=("afr_Latn-aka_Latn" "afr_Latn-bam_Latn" "afr_Latn-nya_Latn" "afr_Latn-run_Latn" "afr_Latn-sot_Latn" "afr_Latn-ssw_Latn" "afr_Latn-tso_Latn")

checkpoints=(
  "/home/jovyan/protasov/LR_Transfer/data/launched_experiments/training_on_Yazva/228a7605-7c4d-4289-9508-d016bbce2351/model_iter_321_epoch_0.pt:Theta_checkpoint"
)

for checkpoint in "${checkpoints[@]}"
do
  checkpoint_path="${checkpoint%%:*}"
  checkpoint_name="${checkpoint##*:}"
  
  for lang_pair in "${lang_pairs[@]}"
  do
    echo "Running python evaluate_mt.py with lang pair: $lang_pair and checkpoint: $checkpoint_path"
    python evaluate_mt.py --lang_pair "$lang_pair" --do_finetune 1 --nickname "${lang_pair}_checkpoint" --experiment_nickname "$checkpoint_name" --base_model_name google/mt5-base --batch_size 4 --from_checkpoint 1 --checkpoint "$checkpoint_path"
  done
done
