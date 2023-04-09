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

nltk.download('punkt')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def get_split(text1):
    l_total = []
    l_partial = []
    if len(text1.split()) // 150 > 0:
        n = len(text1.split()) // 150
    else:
        n = 1
    for w in range(n):
        if w == 0:
            l_partial = text1.split()[:200]
            l_total.append(" ".join(l_partial))
        else:
            l_partial = text1.split()[w * 150:w * 150 + 200]
            l_total.append(" ".join(l_partial))
    return l_total


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


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = RedditCommentDataset(
        doc=df.process_text.to_numpy(),
        targets=df.target.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
    )


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

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

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

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

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)
    
def main():
    annotations = pd.DataFrame(pd.read_csv('annotations.csv'))
    
    df = annotations[['body']].copy()
    df['process_text'] = df['body'].apply(preprocess_text)
    df['process_text'] = df['process_text'].apply(get_split)
    df['target'] = 0
    
    ds = RedditCommentDataset(
        doc=df.process_text.explode().to_numpy(),
        targets=df.target.repeat(df.process_text.apply(len)).to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    n_classes = 2
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=False)

    MAX_LEN = 128
    BATCH_SIZE = 16

    df_train, df_test = train_test_split(ds, test_size=0.4, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.7, random_state=RANDOM_SEED)

    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    model = PostClassifier(n_classes)
    model = model.to(device)

    EPOCHS = 5
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val)
        )

        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
    # Save training history to CSV files for visualization
    train_history = pd.DataFrame({'epoch': list(range(1, EPOCHS + 1)), 'train_loss': history['train_loss'], 'train_acc': history['train_acc']})
    val_history = pd.DataFrame({'epoch': list(range(1, EPOCHS + 1)), 'val_loss': history['val_loss'], 'val_acc': history['val_acc']})

    train_history.to_csv('train_history.csv', index=False)
    val_history.to_csv('val_history.csv', index=False)

    # Plot the training loss and accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_history['epoch'], train_history['train_loss'], label='Train Loss')
    plt.plot(val_history['epoch'], val_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_history['epoch'], train_history['train_acc'], label='Train Accuracy')
    plt.plot(val_history['epoch'], val_history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == '__main__':
    main()


