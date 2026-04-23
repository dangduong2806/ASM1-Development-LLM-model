"""
Advanced finetuning script for generating improved output files.
This script trains with better hyperparameters (more epochs, lower dropout)
and saves results to *-advanced-output.txt files.
"""
from contextlib import nullcontext
import json
import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

from classifier import LlamaZeroShotClassifier, LlamaEmbeddingClassifier
from llama import Llama, load_pretrained
from optimizer import AdamW
from tokenizer import Tokenizer
from tqdm import tqdm
from typing import Optional

# Reuse functions from run_llama
from run_llama import (
    seed_everything, LlamaDataset, create_data, model_eval,
    save_model, write_predictions_to_file
)

TQDM_DISABLE = False


def train_advanced(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    tokenizer = Tokenizer(args.max_sentence_len)
    train_data, num_labels = create_data(args.train, tokenizer, 'train')
    dev_data = create_data(args.dev, tokenizer, 'valid')

    train_dataset = LlamaDataset(train_data, args)
    dev_dataset = LlamaDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'pretrained_model_path': args.pretrained_model_path,
              'num_labels': num_labels,
              'data_dir': '.',
              'option': 'finetune'}

    config = SimpleNamespace(**config)

    model = LlamaEmbeddingClassifier(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_loss = 0
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            b_ids, b_labels, b_sents = batch['token_ids'], batch['labels'], batch['sents']

            b_ids = b_ids.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model(b_ids)
            loss = F.nll_loss(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def test_advanced(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath, weights_only=False)
        config = saved['model_config']
        model = LlamaEmbeddingClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")
        tokenizer = Tokenizer(args.max_sentence_len)
        dev_data = create_data(args.dev, tokenizer, 'valid')
        dev_dataset = LlamaDataset(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

        test_data = create_data(args.test, tokenizer, 'test')
        test_dataset = LlamaDataset(test_data, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

        dev_acc, dev_f1, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device)
        test_acc, test_f1, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device)

        write_predictions_to_file("dev", args.dev_out, dev_acc, dev_pred, dev_sents)
        write_predictions_to_file("test", args.test_out, test_acc, test_pred, test_sents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["sst", "cfimdb"])
    parser.add_argument("--pretrained-model-path", type=str, default="stories42M.pt")
    parser.add_argument("--max_sentence_len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    if args.dataset == "sst":
        args.train = "data/sst-train.txt"
        args.dev = "data/sst-dev.txt"
        args.test = "data/sst-test.txt"
        args.dev_out = "sst-dev-advanced-output.txt"
        args.test_out = "sst-test-advanced-output.txt"
        if args.batch_size is None:
            args.batch_size = 16
    else:
        args.train = "data/cfimdb-train.txt"
        args.dev = "data/cfimdb-dev.txt"
        args.test = "data/cfimdb-test.txt"
        args.dev_out = "cfimdb-dev-advanced-output.txt"
        args.test_out = "cfimdb-test-advanced-output.txt"
        if args.batch_size is None:
            args.batch_size = 4

    args.filepath = f'{args.dataset}-advanced-{args.epochs}-{args.lr}.pt'

    print(f"Advanced finetuning args: {vars(args)}")
    seed_everything(args.seed)

    # Train
    train_advanced(args)

    # Evaluate
    test_advanced(args)
