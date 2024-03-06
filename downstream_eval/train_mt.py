import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from datasets import load_dataset
from tqdm import tqdm
from torchmetrics.functional.text import chrf_score
from torchmetrics.text import BLEUScore
import random
import numpy as np
import os
import argparse


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Define a function to encode the input and target texts
def encode(texts, tokenizer, max_length=512, padding="max_length", truncation=True, return_tensors="pt"):
  return tokenizer(texts, max_length=max_length, padding=padding, truncation=truncation, return_tensors=return_tensors)

# Define a function to train the model on one epoch
def train_epoch(model, optimizer, tokenizer, criterion, train_loader, device,  input_lang, target_lang, epoch):
    # Set the model to training mode
    model.train()
    # Initialize the running loss
    running_loss = 0.0
    # Initialize the progress bar
    pbar = tqdm(train_loader, unit="batch")
    # Loop over the batches
    for _, data in enumerate(pbar):
        # Get the input and target texts from the batch
        input_texts = data['translation'][input_lang]
        target_texts = data['translation'][target_lang]
        # Encode the input and target texts
        input_ids = encode(input_texts, tokenizer)['input_ids'].to(device)
        target_ids = encode(target_texts, tokenizer)['input_ids'].to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(input_ids=input_ids, labels=target_ids)
        # Get the loss and logits from the outputs
        loss = outputs.loss
        logits = outputs.logits
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() # update loss
        
        # Update the progress bar
        pbar.set_description(f"Epoch {epoch+1}")
        pbar.set_postfix(loss=loss.item())
    # Return the average loss
    return running_loss / len(train_loader)

# Define a function to evaluate the model on the test set
def evaluate(model, criterion, tokenizer, eval_loader, device, input_lang, target_lang):
    bleu_evaluator = BLEUScore()
    # Set the model to evaluation mode
    model.eval()
    # Initialize the running loss, and chrF
    running_loss = 0.0
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
            # Get the loss and logits from the outputs
            loss = outputs.loss
            logits = outputs.logits
            # Decode the logits to get the predicted texts
            pred_ids = torch.argmax(logits, dim=-1)
            pred_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_ids]            
            # Calculate chrF for the batch
            chrf = chrf_score(pred_texts, target_texts)
            bleu_score_out = float(bleu_evaluator(pred_texts, [[i] for i in target_texts]))
            # Update the running loss, and chrF
            # running_loss += loss.detach().item()
            # running_chrf += chrf.detach().item()
            running_loss += loss.item()
            running_chrf += chrf.item()
            running_bleu += bleu_score_out
            # Update the progress bar
            pbar.set_description(f"Evaluation")
            pbar.set_postfix(loss=loss.item(), chrf=chrf.item(), bleu=bleu_score_out)
            # Return the average loss, and chrF
    return running_loss / len(eval_loader), running_chrf / len(eval_loader), running_bleu / len(eval_loader)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lang_pair", type=str, help="nnlb valid language pair", default="pol_Latn-ukr_Cyrl", required=True
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size (8 or 16)"
    )
    parser.add_argument(
        "--epoch_num", type=int, default=5, help="number of epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="learning rate"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="learning rate"
    )
    parser.add_argument(
        "--short_run", type=int, default=1, help="run training using small dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="your model name or its path", 
        default="google/mt5-small"
    )

    args = parser.parse_args()
    
    random_seed = args.seed
    input_lang_pair = args.lang_pair
    # Define the hyperparameters
    batch_size = args.batch_size
    num_epochs = args.epoch_num
    learning_rate = args.learning_rate
    model_name = args.model_name
    
    input_lang, target_lang = input_lang_pair.split('-')
    
    set_random_seed(random_seed)
    
    dataset = load_dataset("allenai/nllb", input_lang_pair, ignore_verifications=True)
    
    if args.short_run:
        # Limit dataset for testing purposes, comment out to use full size
        dataset = dataset['train'].train_test_split(test_size=0.005, shuffle=True, seed=random_seed)
        dataset = dataset['test']
        dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=random_seed)
    else:
        # Uncomment to use full size dataset
        dataset = dataset['train'].train_test_split(test_size=0.1, shuffle=True, seed=random_seed)
    
    dataset2 = dataset['test'].train_test_split(test_size=0.5, shuffle=True, seed=random_seed)
    
    # Split the dataset into train and test sets
    train_set = dataset["train"]
    eval_set = dataset2["train"]
    test_set = dataset2["test"]
    
    # Define the model and tokenizer
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    
    # Move the model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Create data loaders for the train and test sets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    
    bleu_evaluator = BLEUScore()
    
    # Start main training
    for epoch in range(num_epochs):
        # Train the model on one epoch
        train_loss = train_epoch(model, optimizer, tokenizer, criterion, train_loader, device, input_lang, target_lang, epoch)
        # Evaluate the model on the test set
        eval_loss, eval_chrf, eval_bleu = evaluate(model, criterion, tokenizer, eval_loader, device, input_lang, target_lang)
        # Print the statistics for the epoch
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, Eval chrF: {eval_chrf:.4f}, Eval Bleu: {eval_bleu:.4f}")
     
    # Run testing
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    test_loss, test_chrf, test_bleu = evaluate(model, criterion, tokenizer, test_loader, device, input_lang, target_lang)
    
    print(f'Test Loss: {test_loss:.4f}, Test chrF: {test_chrf:.4f}, Test Bleu: {test_bleu:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), "mt5_model.pth")
    
    # Model will be saved in the current working directory