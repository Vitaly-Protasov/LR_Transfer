import argparse
import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from torchmetrics.functional.text import chrf_score
from torchmetrics.text import BLEUScore
from tqdm import tqdm
from train_mt import encode, evaluate, set_random_seed, train_epoch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer


def evaluate_zero_shot(model, tokenizer, eval_loader, device):
    bleu_evaluator = BLEUScore()
    # Set the model to evaluation mode
    model.eval()
    # Initialize the running bleu, and chrF
    running_chrf = 0.0
    running_bleu = 0.0
    # Initialize the progress bar
    pbar = tqdm(eval_loader, unit="batch")
    # Loop over the batches
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            # Get the input and target texts from the batch
            input_texts = batch["translation"][input_lang]
            target_texts = batch["translation"][target_lang]
            # Encode the input and target texts
            input_ids = encode(input_texts, tokenizer)["input_ids"].to(device)
            target_ids = encode(target_texts, tokenizer)["input_ids"].to(device)
            # Forward pass
            outputs = model(input_ids=input_ids, labels=target_ids)
            # Get the logits from the outputs
            logits = outputs.logits
            # Decode the logits to get the predicted texts
            pred_ids = torch.argmax(logits, dim=-1)
            pred_texts = [
                tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_ids
            ]
            # Calculate chrF and blue for the batch
            chrf = chrf_score(pred_texts, target_texts)
            bleu_score_out = float(
                bleu_evaluator(pred_texts, [[i] for i in target_texts])
            )
            running_chrf += chrf.item()
            running_bleu += bleu_score_out
            # Update the progress bar
            pbar.set_description(f"Evaluation")
            pbar.set_postfix(chrf=chrf.item(), bleu=bleu_score_out)
            # Return the average bleu and chrF
    return running_chrf / len(eval_loader), running_bleu / len(eval_loader)


def export_results(
    checkpoint_name,
    model_name,
    lang_pair,
    eval_chrf,
    eval_bleu,
    nickname,
    save_dir,
    eval_only_bool,
):
    out_json = {
        "checkpoint_name": checkpoint_name,
        "base_model": model_name,
        "language_pair": lang_pair,
        "only_evaluation": eval_only_bool,
        "metrics": {"chrf": eval_chrf, "bleu": eval_bleu},
    }
    os.makedirs(f"experiment_results/{save_dir}", exist_ok=True)
    with open(
        f"experiment_results/{save_dir}/experiment_results_{nickname}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(out_json, f, ensure_ascii=False, indent=4)

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="path to your checkpoint",
        default="checkpoints/mt5_model.pth",
        required=False,
    )
    parser.add_argument(
        "--lang_pair",
        type=str,
        help="nnlb valid language pair",
        default="pol_Latn-ukr_Cyrl",
        required=True,
    )
    parser.add_argument(
        "--nickname",
        type=str,
        help="nickname for your experiement",
        default="lang_test_run",
    )
    parser.add_argument(
        "--experiment_nickname",
        type=str,
        help="nickname for your experiement",
        default="exp_test_run",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size (8 or 16)"
    )
    parser.add_argument("--epoch_num", type=int, default=5, help="number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="learning rate"
    )
    parser.add_argument("--seed", type=int, default=42, help="learning rate")
    # parser.add_argument(
    #     "--short_run", type=int, default=1, help="run training using small dataset"
    # )
    parser.add_argument(
        "--do_finetune", type=int, default=0, help="whether to finetune"
    )
    parser.add_argument(
        "--from_checkpoint", type=int, default=0, help="whether to use checkpoint"
    )
    parser.add_argument(
        "--how_many_samples",
        type=int,
        default=60000,
        help="how many samples to use out of all the dataset",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        help="your model name or its path",
        default="google/mt5-small",
    )

    args = parser.parse_args()

    # Define hyperparametrs
    batch_size = args.batch_size
    num_epochs = args.epoch_num
    learning_rate = args.learning_rate

    # Define important variables
    model_name = args.base_model_name  # name for the base model
    checkpoint = args.checkpoint  # name of your local checkpoint
    lang_pair = args.lang_pair  # language pair from NLLB
    random_seed = args.seed
    do_finetune = args.do_finetune
    from_pt = args.from_checkpoint
    # short_run = args.short_run
    nickname = args.nickname
    num_samples = args.how_many_samples
    experiment_save_dir = args.experiment_nickname

    set_random_seed(random_seed)

    dataset = load_dataset("allenai/nllb", lang_pair, ignore_verifications=True)
    input_lang, target_lang = lang_pair.split("-")

    # print(dataset)
    # print(len(dataset['train'])*0.2)

    if len(dataset["train"]) >= num_samples:
        target_percentage = round(num_samples / len(dataset["train"]), 3)
        dataset = dataset["train"].train_test_split(
            test_size=target_percentage, shuffle=True, seed=random_seed
        )
        dataset = dataset["test"]
        dataset = dataset.train_test_split(
            test_size=0.16, shuffle=True, seed=random_seed
        )
    else:
        dataset = dataset["train"].train_test_split(
            test_size=0.1, shuffle=True, seed=random_seed
        )

    ln_dataset_train = len(dataset["train"])

    print(f"\nThe following number of samples will be used: {ln_dataset_train}")
    #     if short_run:
    #         dataset = dataset['train'].train_test_split(test_size=0.2, shuffle=True, seed=random_seed)
    #         dataset = dataset['test']
    #         dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=random_seed)
    #     else:
    #         dataset = dataset['train'].train_test_split(test_size=0.1, shuffle=True, seed=random_seed)

    dataset2 = dataset["test"].train_test_split(
        test_size=0.5, shuffle=True, seed=random_seed
    )

    # Split the dataset into train and test sets
    train_set = dataset["train"]
    eval_set = dataset2["train"]
    test_set = dataset2["test"]

    # Define the model and tokenizer

    if do_finetune and not from_pt:
        model = MT5ForConditionalGeneration.from_pretrained(model_name)
    elif do_finetune and from_pt:
        model = MT5ForConditionalGeneration.from_pretrained(model_name)
        model.load_state_dict(torch.load(checkpoint))
    else:
        model = MT5ForConditionalGeneration.from_pretrained(model_name)
        model.load_state_dict(torch.load(checkpoint))
    tokenizer = MT5Tokenizer.from_pretrained(model_name)

    # Move the model to GPU
    device = "cuda"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    model.to(device)

    if do_finetune:
        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Define the loss function
        criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        # Create data loaders for the train and test sets
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=False
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_set, batch_size=batch_size, shuffle=False
        )
    # elif short_run:
    #     dataset_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    else:
        dataset_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=False
        )

    if do_finetune:
        print("\nStarted finetuning!")
        only_eval = False
        # Start main training
        for epoch in range(num_epochs):
            # Train the model on one epoch
            train_loss = train_epoch(
                model,
                optimizer,
                tokenizer,
                criterion,
                train_loader,
                device,
                input_lang,
                target_lang,
                epoch,
            )
            # Evaluate the model on the test set
            eval_loss, eval_chrf, eval_bleu = evaluate(
                model,
                criterion,
                tokenizer,
                eval_loader,
                device,
                input_lang,
                target_lang,
            )
            # Print the statistics for the epoch
            print(
                f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, Eval chrF: {eval_chrf:.4f}, Eval Bleu: {eval_bleu:.4f}"
            )

        # Run testing
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False
        )
        test_loss, test_chrf, test_bleu = evaluate(
            model, criterion, tokenizer, test_loader, device, input_lang, target_lang
        )

        print(
            f"Test Loss: {test_loss:.4f}, Test chrF: {test_chrf:.4f}, Test Bleu: {test_bleu:.4f}"
        )
        os.makedirs(f"finetuned_checkpoints/{experiment_save_dir}", exist_ok=True)
        model_name_out = (
            f"finetuned_checkpoints/{experiment_save_dir}/mt5_model_{lang_pair}.pth"
        )
        if from_pt:
            curr_name_checkpoint = checkpoint
        else:
            curr_name_checkpoint = model_name
        # Save the model
        torch.save(
            model.state_dict(), model_name_out
        )  # Model will be saved in the current working directory

        nickname = nickname + "_finetune"
        export_results(
            curr_name_checkpoint,
            model_name,
            lang_pair,
            test_chrf,
            test_bleu,
            nickname,
            experiment_save_dir,
            only_eval,
        )
    else:
        only_eval = True
        print("\nStarted zero-shot evaluation!")
        eval_chrf, eval_bleu = evaluate_zero_shot(
            model, tokenizer, dataset_loader, device
        )

        print(f"\nZero-shot chrF: {eval_chrf:.4f}, Zero-shot Bleu: {eval_bleu:.4f}")

        checkpoint_name = checkpoint.split("/")[-1]

        nickname = nickname + "_zeroshot"
        export_results(
            checkpoint_name,
            model_name,
            lang_pair,
            eval_chrf,
            eval_bleu,
            nickname,
            experiment_save_dir,
            only_eval,
        )
