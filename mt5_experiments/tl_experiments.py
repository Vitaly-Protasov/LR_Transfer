from transformers import (
    MT5ForConditionalGeneration,
    T5Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
import torch
from tqdm.auto import tqdm, trange
from datasets import load_dataset
import os
import random
import json
from pathlib import Path
import numpy as np
from flax.training.common_utils import shard
from enum import Enum
from typing import Optional, Tuple, List
import uuid

import tl_utils


class mt5PerplexityExperiments:
    def __init__(
        self,
        model_id: Enum = "google/mt5-base",
        device: Enum = "cuda:0",
    ):
        self.device = device
        self.model = MT5ForConditionalGeneration.from_pretrained(model_id).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_id)

        self.log_dict = {}

    def get_tokenized_dataset(self, datasets, column_name):
        max_seq_length = min(self.max_seq_length, self.tokenizer.model_max_length)
        column_names = datasets[column_name].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        tokenized_datasets = datasets.map(
            lambda x: tl_utils.tokenize_function(
                x, tokenizer=self.tokenizer, text_column_name=text_column_name
            ),
            batched=True,
            num_proc=self.num_proc,
            remove_columns=column_names,
        )
        (
            expanded_inputs_length,
            targets_length,
        ) = tl_utils.compute_input_and_target_lengths(
            inputs_length=self.max_seq_length,
            noise_density=self.mlm_probability,
            mean_noise_span_length=self.mean_noise_span_length,
        )

        data_collator = tl_utils.FlaxDataCollatorForT5MLM(
            tokenizer=self.tokenizer,
            noise_density=self.mlm_probability,
            mean_noise_span_length=self.mean_noise_span_length,
            input_length=max_seq_length,
            target_length=targets_length,
            pad_token_id=self.model.config.pad_token_id,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

        tokenized_datasets = tokenized_datasets.map(
            lambda x: tl_utils.group_texts(
                x, expanded_inputs_length=expanded_inputs_length
            ),
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
        lr_languages_to_test: List[str] = [],
        save_checkpoints: bool = False,
    ):
        log_params = {
            "train_valid_dir": train_valid_dir,
            "train_size": train_size,
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "num_warmup_steps": num_warmup_steps,
            "weight_decay": weight_decay,
            "betas": betas,
            "max_seq_length": max_seq_length,
            "per_device_batch_size": per_device_batch_size,
            "mlm_probability": mlm_probability,
            "mean_noise_span_length": mean_noise_span_length,
            "num_proc": num_proc,
        }
        random_seed = uuid.uuid4()
        self.save_folder = f"launched_experiments/training_on_{Path(train_valid_dir).name}/{random_seed}"
        self.max_seq_length = max_seq_length
        self.per_device_batch_size = per_device_batch_size
        self.mlm_probability = mlm_probability
        self.mean_noise_span_length = mean_noise_span_length
        self.num_proc = num_proc

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        params_filename = Path(self.save_folder, "params.json")
        log_filename = Path(self.save_folder, "log_results.txt")
        log_errors = Path(self.save_folder, "log_errors.txt")
        new_log_path = Path(self.save_folder, "new_log.json")

        with open(params_filename, "w+") as outfile:
            json.dump(log_params, outfile, indent=4)

        train_val_paths = [
            str(Path(train_valid_dir, i)) for i in os.listdir(train_valid_dir)
        ]
        dataset = load_dataset("text", data_files=train_val_paths, split="train")

        data_indices = np.arange(len(dataset))
        random.shuffle(data_indices)
        dataset_limit = min(len(dataset), max_dataset_len)
        cutted_dataset = dataset.select(data_indices[:dataset_limit])
        datasets = cutted_dataset.train_test_split(test_size=1 - train_size)

        train_tokenized_datasets, train_data_collator = self.get_tokenized_dataset(
            datasets, "train"
        )
        num_train_samples = len(train_tokenized_datasets["train"])
        train_batch_idx = tl_utils.generate_batch_splits(
            np.arange(num_train_samples), self.per_device_batch_size
        )

        num_train_steps = (
            len(train_tokenized_datasets["train"])
            // self.per_device_batch_size
            * n_epochs
        )

        val_tokenized_datasets, val_data_collator = self.get_tokenized_dataset(
            datasets, "test"
        )
        num_val_samples = len(val_tokenized_datasets["test"])
        val_batch_idx = tl_utils.generate_batch_splits(
            np.arange(num_val_samples), self.per_device_batch_size
        )

        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        self.log_dict = {"train": [], "val": [], "test": {}}
        for epoch in trange(n_epochs):
            # ======================== Training ================================
            train_losses_epoch = []

            step = int(len(train_batch_idx) * 0.05)
            for i, batch_idx in tqdm(
                enumerate(train_batch_idx),
                desc="Training...",
                total=len(train_batch_idx),
            ):
                with open(str(new_log_path), "w") as outfile:
                    json.dump(self.log_dict, outfile)

                tl_utils.clear_memory()

                self.model.train()
                f = open(log_filename, "a+")
                f_error = open(log_errors, "a+")

                samples = [
                    train_tokenized_datasets["train"][int(idx)] for idx in batch_idx
                ]
                model_inputs = train_data_collator(samples)
                model_inputs = shard(model_inputs.data)

                input_ids = torch.LongTensor(model_inputs["input_ids"]).to(self.device)
                labels = torch.LongTensor(model_inputs["labels"]).to(self.device)

                input_ids_size = input_ids.size()
                labels_size = labels.size()
                input_ids = input_ids.reshape(
                    [input_ids_size[0], input_ids_size[1] * input_ids_size[2]]
                )
                labels = labels.reshape(
                    [labels_size[0], labels_size[1] * labels_size[2]]
                )

                optimizer.zero_grad()
                loss = self.model(input_ids=input_ids, labels=labels)

                train_losses_epoch.append(loss.loss.item())
                loss.loss.backward()
                optimizer.step()
                scheduler.step()

                # ======================== Evaluating ==============================
                if i % step == 0 and i > 0:
                    perp_train = np.exp(np.mean(train_losses_epoch))
                    train_msg = f"\nTRAIN ITERATION: {i}\t FOR {train_valid_dir} \t Perplexity = {perp_train}\n"
                    print(train_msg)
                    f.write(train_msg)

                    self.log_dict["train"].append(perp_train)

                    self.model.eval()
                    with torch.no_grad():
                        tl_utils.clear_memory()

                        val_losses_epoch = []
                        for batch_idx in tqdm(val_batch_idx, desc="Validation..."):
                            samples = [
                                val_tokenized_datasets["test"][int(idx)]
                                for idx in batch_idx
                            ]
                            model_inputs = val_data_collator(samples)
                            model_inputs = shard(model_inputs.data)

                            input_ids = torch.LongTensor(model_inputs["input_ids"]).to(
                                self.device
                            )
                            labels = torch.LongTensor(model_inputs["labels"]).to(
                                self.device
                            )

                            input_ids_size = input_ids.size()
                            labels_size = labels.size()
                            input_ids = input_ids.reshape(
                                [
                                    input_ids_size[0],
                                    input_ids_size[1] * input_ids_size[2],
                                ]
                            )
                            labels = labels.reshape(
                                [labels_size[0], labels_size[1] * labels_size[2]]
                            )
                            loss = self.model(input_ids=input_ids, labels=labels)
                            val_losses_epoch.append(loss.loss.item())

                        perp_val = np.exp(np.mean(val_losses_epoch))
                        val_msg = f"\nVALIDATION ITERATION: {i}\t FOR {train_valid_dir} \t Perplexity = {perp_val}\n"
                        print(val_msg)
                        f.write(val_msg)
                        f.close()

                        if save_checkpoints and perp_val < min(
                            self.log_dict["val"], default=1000000
                        ):
                            torch.save(
                                self.model.state_dict(),
                                Path(self.save_folder, "best_model.pt"),
                            )

                        self.log_dict["val"].append(perp_val)

                        tl_utils.clear_memory()

                        for lr_lang in tqdm(lr_languages_to_test):
                            lang_folder_path = str(
                                Path(Path(train_valid_dir).parent, lr_lang)
                            )
                            try:
                                if lang_folder_path not in self.log_dict["test"]:
                                    self.log_dict["test"][lang_folder_path] = []
                                self.log_dict["test"][lang_folder_path].append(
                                    self.testing(lang_folder_path)
                                )
                            except:
                                f_error.write(
                                    f"Something went wrong during processing: {lang_folder_path}\n"
                                )

    def testing(
        self,
        test_dir: os.PathLike,
        max_seq_length: int = 256,
        per_device_batch_size: int = 64,
        mlm_probability: float = 0.15,
        mean_noise_span_length: int = 3,
        num_proc: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
    ):
        self.max_seq_length = max_seq_length
        self.per_device_batch_size = per_device_batch_size
        self.mlm_probability = mlm_probability
        self.mean_noise_span_length = mean_noise_span_length
        self.num_proc = num_proc

        if checkpoint_path is not None:
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )

        test_paths = [str(Path(test_dir, i)) for i in os.listdir(test_dir)]
        datasets = load_dataset("text", data_files=test_paths)

        test_tokenized_datasets, test_data_collator = self.get_tokenized_dataset(
            datasets, "train"
        )
        num_test_samples = len(test_tokenized_datasets["train"])
        test_batch_idx = tl_utils.generate_batch_splits(
            np.arange(num_test_samples), self.per_device_batch_size
        )
        self.model.eval()
        with torch.no_grad():
            tl_utils.clear_memory()

            test_losses = []
            for batch_idx in tqdm(test_batch_idx, desc="Testing..."):
                samples = [
                    test_tokenized_datasets["train"][int(idx)] for idx in batch_idx
                ]
                model_inputs = test_data_collator(samples)
                model_inputs = shard(model_inputs.data)

                input_ids = torch.LongTensor(model_inputs["input_ids"]).to(self.device)
                labels = torch.LongTensor(model_inputs["labels"]).to(self.device)

                input_ids_size = input_ids.size()
                labels_size = labels.size()
                input_ids = input_ids.reshape(
                    [input_ids_size[0], input_ids_size[1] * input_ids_size[2]]
                )
                labels = labels.reshape(
                    [labels_size[0], labels_size[1] * labels_size[2]]
                )
                loss = self.model(input_ids=input_ids, labels=labels)
                test_losses.append(loss.loss.item())

            perp_test = np.exp(np.mean(test_losses))
            test_msg = f"\nTEST: For {test_dir} \t Perplexity = {perp_test}\n"
            print(test_msg)
        return perp_test
