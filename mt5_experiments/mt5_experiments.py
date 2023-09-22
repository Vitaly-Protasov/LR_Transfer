from transformers import (
    MT5ForConditionalGeneration,
    T5Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
import torch
from datasets import load_dataset
import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm, trange
from flax.training.common_utils import shard
from enum import Enum
from typing import Optional, Tuple
from fire import Fire
from math import floor
import uuid

import mt5_utils


class mt5PerplexityExperiments:

    def __init__(
        self,
        model_id: Enum = 'google/mt5-base',
        device: Enum = 'cuda:0',
    ):
        self.device = device
        self.model = MT5ForConditionalGeneration.from_pretrained(model_id).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_id)

    def get_tokenized_dataset(self, datasets, column_name):
        max_seq_length = min(self.max_seq_length, self.tokenizer.model_max_length)
        column_names = datasets[column_name].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        
        tokenized_datasets = datasets.map(
            lambda x: mt5_utils.tokenize_function(x, tokenizer=self.tokenizer, text_column_name=text_column_name),
            batched=True,
            num_proc=self.num_proc,
            remove_columns=column_names
        )
        expanded_inputs_length, targets_length = mt5_utils.compute_input_and_target_lengths(
            inputs_length=self.max_seq_length,
            noise_density=self.mlm_probability,
            mean_noise_span_length=self.mean_noise_span_length,
        )

        data_collator = mt5_utils.FlaxDataCollatorForT5MLM(
            tokenizer=self.tokenizer,
            noise_density=self.mlm_probability,
            mean_noise_span_length=self.mean_noise_span_length,
            input_length=max_seq_length,
            target_length=targets_length,
            pad_token_id=self.model.config.pad_token_id,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

        tokenized_datasets = tokenized_datasets.map(
            lambda x: mt5_utils.group_texts(x, expanded_inputs_length=expanded_inputs_length),
            batched=True,
            num_proc=self.num_proc,
        )
        return tokenized_datasets, data_collator


    def training(
        self,
        train_valid_dir: os.PathLike,
        max_dataset_len: int = 500000,
        train_size: float = 0.9,
        n_epochs: int = 5,
        learning_rate: float = 0.005,
        num_warmup_steps: int = 2000,
        weight_decay: float = 0.001,
        betas: Tuple[float, float] = [0.9, 0.999],
        max_seq_length: int = 256,
        per_device_batch_size: int = 64,
        mlm_probability: float = 0.15,
        mean_noise_span_length: int = 3,
        num_proc: Optional[int] = None,
    ):
        self.max_seq_length = max_seq_length
        self.per_device_batch_size = per_device_batch_size
        self.mlm_probability = mlm_probability
        self.mean_noise_span_length = mean_noise_span_length
        self.num_proc = num_proc

        log_params = {
            "train_valid_dir":train_valid_dir,
            "train_size":train_size,
            "n_epochs":n_epochs,
            "learning_rate":learning_rate,
            "num_warmup_steps":num_warmup_steps,
            "weight_decay":weight_decay,
            "betas":betas,
            "max_seq_length":max_seq_length,
            "per_device_batch_size":per_device_batch_size,
            "mlm_probability":mlm_probability,
            "mean_noise_span_length":mean_noise_span_length,
            "num_proc":num_proc
        }
        random_seed = uuid.uuid4()
        save_folder = f'mt5_experiments/training_on_{Path(train_valid_dir).name}/{random_seed}'

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        params_filename = Path(save_folder, "params.json")
        log_filename = Path(save_folder, "log_results.txt")
        with open(params_filename, "w") as outfile:
            json.dump(log_params, outfile, indent=4)

        train_val_paths = [str(Path(train_valid_dir, i)) for i in os.listdir(train_valid_dir)]
        dataset = load_dataset('text', data_files=train_val_paths, split='train')

        dataset_limit = min(len(dataset), max_dataset_len)
        data_indices = np.random.choice(len(dataset), dataset_limit)
        cutted_dataset = dataset.select(data_indices)
        datasets = cutted_dataset.train_test_split(test_size=1-train_size)
        column_name = 'train'

        train_tokenized_datasets, train_data_collator = self.get_tokenized_dataset(datasets, column_name)
        num_train_samples = len(train_tokenized_datasets[column_name])
        train_batch_idx = mt5_utils.generate_batch_splits(
            np.arange(num_train_samples),
            self.per_device_batch_size
            )
        
        num_train_steps = len(train_tokenized_datasets["train"]) // self.per_device_batch_size * n_epochs
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay = weight_decay,
            betas = betas
            )

        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps
                )
        for epoch in trange(n_epochs):
            # ======================== Training ================================
            train_losses_epoch = []

            step = int(len(train_batch_idx) * 0.05)
            for i, batch_idx in tqdm(enumerate(train_batch_idx), desc='Training...', leave=True):
                self.model.train()
                f = open(log_filename, 'a+')
       
                samples = [train_tokenized_datasets["train"][int(idx)] for idx in batch_idx]
                model_inputs = train_data_collator(samples)
                model_inputs = shard(model_inputs.data)

                input_ids = torch.LongTensor(model_inputs['input_ids']).to(self.device)
                # decoder_input_ids = torch.LongTensor(model_inputs['decoder_input_ids']).to(self.device)
                labels = torch.LongTensor(model_inputs['labels']).to(self.device)

                loss = self.model(
                    input_ids=torch.squeeze(input_ids, 0),
                    labels=torch.squeeze(labels, 0)
                )
                train_losses_epoch.append(loss.loss.item())
                loss.loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # ======================== Evaluating ==============================
                if i % step == 0 and i > 0:
                    train_msg = f'TRAIN ITERATION: {i}\t FOR {train_valid_dir} \t Perplexity = {np.exp(np.mean(train_losses_epoch))}\n'
                    print(train_msg)
                    f.write(train_msg)

                    self.model.eval()

                    with torch.no_grad():
                        column_name = 'test'
                        val_tokenized_datasets, val_data_collator = self.get_tokenized_dataset(datasets, column_name)
                        num_val_samples = len(val_tokenized_datasets[column_name])
                        val_batch_idx = mt5_utils.generate_batch_splits(
                            np.arange(num_val_samples),
                            self.per_device_batch_size
                            )
                        val_losses_epoch = []
                        for batch_idx in tqdm(val_batch_idx, desc='Validation...', leave=True):
                            samples = [val_tokenized_datasets[column_name][int(idx)] for idx in batch_idx]
                            model_inputs = val_data_collator(samples)
                            model_inputs = shard(model_inputs.data)

                            input_ids = torch.LongTensor(model_inputs['input_ids']).to(self.device)
                            # decoder_input_ids = torch.LongTensor(model_inputs['decoder_input_ids']).to(self.device)
                            labels = torch.LongTensor(model_inputs['labels']).to(self.device)

                            loss = self.model(
                                input_ids=torch.squeeze(input_ids, 0),
                                labels=torch.squeeze(labels, 0)
                            )
                            val_losses_epoch.append(loss.loss.item())

                        val_msg = f'VALIDATION ITERATION: {i}\t FOR {train_valid_dir} \t Perplexity = {np.exp(np.mean(val_losses_epoch))}\n'
                        print(val_msg)
                        f.write(val_msg)
                        f.close()
                        torch.save(self.model.state_dict(), Path(save_folder, f'epoch_{epoch}_iteration_{i}.pt'))

    def testing(
        self,
        test_dir: os.PathLike,
        max_seq_length: int = 256,
        per_device_batch_size: int = 64,
        mlm_probability: float = 0.15,
        mean_noise_span_length: int = 3,
        num_proc: Optional[int] = None,
        checkpoint_path: Optional[str] = None
    ):
        self.max_seq_length = max_seq_length
        self.per_device_batch_size = per_device_batch_size
        self.mlm_probability = mlm_probability
        self.mean_noise_span_length = mean_noise_span_length
        self.num_proc = num_proc
        
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
    
        test_paths = [str(Path(test_dir, i)) for i in os.listdir(test_dir)]
        datasets = load_dataset('text', data_files=test_paths)

        self.model.eval()

        with torch.no_grad():
            column_name = 'train'
            test_tokenized_datasets, test_data_collator = self.get_tokenized_dataset(datasets, column_name)
            num_test_samples = len(test_tokenized_datasets[column_name])
            test_batch_idx = mt5_utils.generate_batch_splits(
                np.arange(num_test_samples),
                self.per_device_batch_size
                )
            test_losses = []
            for batch_idx in tqdm(test_batch_idx, desc='Testing...', leave=True):
                samples = [test_tokenized_datasets[column_name][int(idx)] for idx in batch_idx]
                model_inputs = test_data_collator(samples)
                model_inputs = shard(model_inputs.data)

                input_ids = torch.LongTensor(model_inputs['input_ids']).to(self.device)
                # decoder_input_ids = torch.LongTensor(model_inputs['decoder_input_ids']).to(self.device)
                labels = torch.LongTensor(model_inputs['labels']).to(self.device)

                loss = self.model(
                    input_ids=torch.squeeze(input_ids, 0),
                    labels=torch.squeeze(labels, 0)
                )
                test_losses.append(loss.loss.item())
            
            test_msg = (f'TEST: For {test_dir} \t Perplexity = {np.exp(np.mean(test_losses))}\n')
            print(test_msg)
            return np.exp(np.mean(test_losses))


def main(
    train_valid_dir: Optional[os.PathLike] = None,
    max_dataset_len: int = 500000,
    train_size: float = 0.9,
    n_epochs: int = 5,
    learning_rate: float = 0.005,
    num_warmup_steps: int = 2000,
    weight_decay: float = 0.001,
    betas: Tuple[float, float] = [0.9, 0.999],
    test_dir: Optional[os.PathLike] = None,
    model_id: Enum = 'google/mt5-base',
    device: Enum = 'cuda:0',
    max_seq_length: int = 256,
    per_device_batch_size: int = 64,
    mlm_probability: float = 0.15,
    mean_noise_span_length: int = 3,
    num_proc: Optional[int] = None
):
    initialize_experiments = mt5PerplexityExperiments(
        model_id,
        device,
    )
    if train_valid_dir is not None:
        initialize_experiments.training(
            train_valid_dir,
            max_dataset_len,
            train_size,
            n_epochs,
            learning_rate,
            num_warmup_steps,
            weight_decay,
            betas,
            max_seq_length,
            per_device_batch_size,
            mlm_probability,
            mean_noise_span_length,
            num_proc
        )

    if test_dir is not None:
        initialize_experiments.testing(
            test_dir,
            max_seq_length,
            per_device_batch_size,
            mlm_probability,
            mean_noise_span_length,
            num_proc
        )

if __name__ == "__main__":
    Fire(main)
