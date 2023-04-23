import os
import copy
import json
import transformers
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import notebook as notebook_tqdm
import nltk
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from torch.optim import AdamW
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt

#Random seed for later
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

#Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Function for removing links, numbers, punctuation from comment text 
def preprocess_text(text):
    
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

#Function to split the comments into lists of strings no more than 200 tokens and no less than 150 tokens long
def get_split(text1):
    
    l_total = []
    l_partial = []

    if len(text1.split())//150 >0:
        n = len(text1.split())//150
    else: 
        n = 1
    for w in range(n):
        if w == 0:
            l_partial = text1.split()[:200]
            l_total.append(" ".join(l_partial))
        else:
            l_partial = text1.split()[w*150:w*150 + 200]
            l_total.append(" ".join(l_partial))
    return l_total

#Initialize the Pytorch dataset
class RedditCommentDataset(Dataset):
    
    def __init__(self, doc, targets, tokenizer, max_len):
        self.doc = doc
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.doc)
  
    def __getitem__(self, item):
        doc = str(self.doc[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
                    doc,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                    )

        return {
            'doc_text': doc,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
            }

class PostClassifier(nn.Module):
    
    def __init__(self, n_classes):
        super(PostClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained('roberta-base', return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

#Function to create Pytorch data loaders using previously built Dataset
def create_data_loader(df, tokenizer, max_len, batch_size):
    
    # Preprocess text and create "process_text" column
    df["process_text"] = df["text"].apply(preprocess_text)
    df["docs"] = df["process_text"].apply(get_split)

    # Initialize dataset using processed text and targets
    ds = RedditCommentDataset(
        doc=df.docs.to_numpy(),
        targets=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
    )


#Define function for training loop
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    
    model = model.train()

    losses = []
    correct_predictions = 0

    #Load data in using data loader
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

#Function for validation step
def eval_model(model, data_loader, loss_fn, device, n_examples):
    
    model = model.eval()

    losses = []
    val_preds = []
    true_labels = []
    correct_predictions = 0

    #Validation loop in no gradient mode
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            true_labels.extend(targets.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses), true_labels, val_preds

def compute_class_weights(labels):
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(class_weights, dtype=torch.float).to(device)

def main():
    
    #Define number of classes to predict
    n_classes = 2

    #Initialize tokennizer
    tokenizer=AutoTokenizer.from_pretrained('roberta-base',use_fast=False)

    #Define max length and batch size
    MAX_LEN = 128
    BATCH_SIZE = 16

    # Open the JSONL file and read its contents into a list
    with open('zach_classification.jsonl', 'r') as f:
        jsonl_data = [json.loads(line) for line in f]

    # Convert the list into a pandas DataFrame
    df = pd.DataFrame(jsonl_data)
    gpt_df = pd.read_csv('gpt_gen_labeled.csv')

    # Convert 'accept' and 'reject' labels into 1 and 0, respectively
    df['label'] = df['answer'].apply(lambda x: 1 if x == 'accept' else 0)

    #Train, val, test split
    df_train, df_test = train_test_split(df, test_size=0.4, random_state=RANDOM_SEED)

    df_train = pd.concat([df_train, gpt_df], ignore_index=True)
    df_val, df_test = train_test_split(df_test, test_size=0.7, random_state=RANDOM_SEED)

    #Initialize data loaders
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    data = next(iter(train_data_loader))


    #Initialize model
    model = PostClassifier(n_classes)
    #Pass model to device
    model = model.to(device)

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)

    #Define number of epochs
    EPOCHS = 5

    #Initialize optimizer and define number of training and validation steps
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader) * EPOCHS

    #Initialize learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    #Define loss function
    train_labels = df_train.label.to_numpy()
    class_weights = compute_class_weights(train_labels)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)


    history = defaultdict(list)
    best_accuracy = 0

    #Training and Validation loop
    for epoch in range(EPOCHS):

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss, true_labels, val_preds = eval_model(model, val_data_loader, loss_fn, device, len(df_val))
        

        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        with open('loss.tsv', 'a') as f:
            if epoch == 0:
                f.write('EPOCH\tTIMESTAMP\tBAD_EPOCHS\tLEARNING_RATE\tTRAIN_LOSS\tDEV_LOSS\tDEV_PRECISION\tDEV_RECALL\tDEV_F1\tDEV_ACCURACY\n')
            current_time = datetime.now().strftime("%H:%M:%S")
            learning_rate = scheduler.get_last_lr()[0]
            dev_precision, dev_recall, dev_f1, _ = precision_recall_fscore_support(df_val.label.to_numpy(), val_preds, average='weighted')
            dev_accuracy = val_acc.item()
            f.write(f'{epoch+1}\t{current_time}\t0\t{learning_rate:.4f}\t{train_loss:.8f}\t{val_loss:.8f}\t{dev_precision:.4f}\t{dev_recall:.4f}\t{dev_f1:.4f}\t{dev_accuracy:.4f}\n')

        #Save model if validation accuracy is better than previously best validation accuracy
        if val_acc > best_accuracy:
            torch.save(model, 'best_model.pt')
            best_accuracy = val_acc

    # Evaluate the best model on test data
    test_acc, test_loss, true_labels, test_preds = eval_model(model, test_data_loader, loss_fn, device, len(df_test))

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, test_preds)

    # Calculate classification report
    cr = classification_report(true_labels, test_preds)

    # Print confusion matrix and classification report
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    
if __name__ == '__main__':
    main()



