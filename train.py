
from cProfile import label
from cgitb import text
import json
import os
import random
from re import L
from tabnanny import verbose
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS, OptimizerLRScheduler
from numpy import dtype
from scipy import datasets
from torch import tensor
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    RobertaTokenizer
)
import torch
import lightning.pytorch as pl
import sys
from sklearn.model_selection import train_test_split

MODEL_NAME = 'Salesforce/codet5-base'
N_EPOCHS = 5





class DataPreparation():
    def __init__(self, data):
        self.data = data
        
    def prepare(self):
        texts, labels = [] , []
        for item in self.data['rows']:
            texts.append(item['row']['func_code_string'])
            labels.append(item['row']['func_documentation_string'])
        return train_test_split(texts, labels, test_size=0.5)
        
    def max_lengths(self):
        source_len = 0
        target_len = 0
        for item in self.data['rows']:
            row = item['row']
            if len(row['func_code_tokens']) > source_len:
                source_len = len(row['func_code_tokens'])
            if len(row['func_documentation_tokens']) > target_len:
                target_len = len(row['func_documentation_tokens'])
        
        return source_len, target_len
    
    def tokenize(self, texts, max_length=None):
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
        return tokenizer(
            texts,
            max_length=max_length if max_length is not None else None,
            pad_to_max_length=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )

class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        if encodings['input_ids'].size() == torch.tensor([1,5,512]):
            encodings['input_ids'] = encodings['input_ids'].squeeze(0)
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels['input_ids'][idx]
        }
        return item

    def __len__(self):
        return len(self.labels)

class CodeT5(pl.LightningModule):
    
    def __init__(self) -> None:
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
        
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output.loss , output.logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return AdamW(self.parameters(), lr=0.0001)


def test(x):
    for batch in x:
        sys.exit(batch)

if __name__ == '__main__':
    # 0. Read json file
    with open('data.json','r') as f:
        data = json.load(f)
    
    # 1. Preprocessing
    dp = DataPreparation(data)
    train_texts, val_texts, train_labels, val_labels = dp.prepare()
    source_len, target_length = dp.max_lengths()
    
    train_encodings = dp.tokenize(train_texts)
    train_labels = dp.tokenize(train_labels, target_length)
    val_encodings = dp.tokenize(val_texts, source_len)
    val_labels = dp.tokenize(val_labels, target_length)
    print('1. Preprocessing done.')
    
    # 2. Dataset and Model instantiation
    code_dataset = CodeDataset(train_encodings, train_labels)
    val_dataset = CodeDataset(val_encodings, val_labels)
    dataset_loader = torch.utils.data.DataLoader(code_dataset, batch_size=5, shuffle=True)
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=5, num_workers=7)
    model = CodeT5()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoints',
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )
    
    csv_logger = pl.loggers.CSVLogger(
        save_dir='logs',
        name='code-t5_v1',
    )
    
    trainer = pl.Trainer(
        callbacks=checkpoint_callback,
        max_epochs=N_EPOCHS,
        logger=csv_logger,
        devices=1,
        accelerator='gpu',
        # fast_dev_run=True
    )
    print('2. Dataset and Model instantiation done.')
    print('3. Starting Training script..')
    trainer.fit(model, train_dataloaders=dataset_loader, val_dataloaders=val_dataset_loader)
    
