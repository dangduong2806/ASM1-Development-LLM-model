"""
Hyperparameter search script for Llama2 finetuning.
Optimized for RTX 2050 (4GB VRAM) using:
  - Mixed precision (fp16) to cut VRAM ~50% and speed up 2x
  - In-process execution (no subprocess overhead, model reuse)
  - Gradient accumulation to simulate large effective batch sizes
"""
import time, random, os, csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from types import SimpleNamespace
from tqdm import tqdm

from classifier import LlamaEmbeddingClassifier
from optimizer import AdamW
from tokenizer import Tokenizer
from run_llama import (
    LlamaDataset, create_data, model_eval, 
    write_predictions_to_file, seed_everything
)

# ============================================================
# CONFIG: All experiments to run
# ============================================================
SST_BASE = {
    "train": "data/sst-train.txt", "dev": "data/sst-dev.txt",
    "test": "data/sst-test.txt", "label_names": "data/sst-label-mapping.json",
    "dev_out": "sst-dev-finetuning-output.txt", "test_out": "sst-test-finetuning-output.txt",
}
CFIMDB_BASE = {
    "train": "data/cfimdb-train.txt", "dev": "data/cfimdb-dev.txt",
    "test": "data/cfimdb-test.txt", "label_names": "data/cfimdb-label-mapping.json",
    "dev_out": "cfimdb-dev-finetuning-output.txt", "test_out": "cfimdb-test-finetuning-output.txt",
}

EXPERIMENTS = [
    # SST-5: With fp16 we can push batch_size higher on RTX 2050
    # (name, dataset_base, micro_bs, grad_accum, lr, epochs, dropout)
    ("SST default",       SST_BASE,    32, 2, 2e-5,  5, 0.3),   # effective bs=64
    ("SST lr=5e-5",       SST_BASE,    32, 2, 5e-5,  5, 0.3),
    ("SST lr=1e-4",       SST_BASE,    32, 2, 1e-4,  5, 0.3),
    ("SST lr=1e-5 ep10",  SST_BASE,    32, 2, 1e-5, 10, 0.3),
    ("SST lr=5e-5 ep10",  SST_BASE,    32, 2, 5e-5, 10, 0.3),
    ("SST drop=0.1",      SST_BASE,    32, 2, 2e-5, 10, 0.1),
    ("SST drop=0.2 5e-5", SST_BASE,    32, 2, 5e-5, 10, 0.2),

    # CFIMDB: longer sequences, smaller micro-batch
    ("CFIMDB default",        CFIMDB_BASE, 8, 1, 2e-5,  5, 0.3),  # effective bs=8
    ("CFIMDB lr=5e-5",        CFIMDB_BASE, 8, 1, 5e-5,  5, 0.3),
    ("CFIMDB lr=1e-4",        CFIMDB_BASE, 8, 1, 1e-4,  5, 0.3),
    ("CFIMDB lr=1e-5 ep10",   CFIMDB_BASE, 8, 1, 1e-5, 10, 0.3),
    ("CFIMDB lr=5e-5 ep10",   CFIMDB_BASE, 8, 1, 5e-5, 10, 0.3),
    ("CFIMDB drop=0.1",       CFIMDB_BASE, 8, 1, 2e-5, 10, 0.1),
    ("CFIMDB drop=0.2 5e-5",  CFIMDB_BASE, 8, 1, 5e-5, 10, 0.2),
]

# ============================================================
# TRAINING with mixed precision + gradient accumulation
# ============================================================
def train_with_amp(config_ns, train_dataloader, dev_dataloader, 
                   lr, epochs, micro_bs, grad_accum_steps, device, filepath):
    """Train with fp16 mixed precision and gradient accumulation."""
    model = LlamaEmbeddingClassifier(config_ns)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()
    best_dev_acc = 0
    effective_bs = micro_bs * grad_accum_steps
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_dataloader, desc=f'train-{epoch}', leave=False)
        for step, batch in enumerate(pbar):
            b_ids = batch['token_ids'].to(device)
            b_labels = batch['labels'].to(device)
            
            # Forward with mixed precision
            with autocast():
                logits = model(b_ids)
                loss = F.nll_loss(logits, b_labels.view(-1), reduction='sum') / effective_bs
            
            # Backward with gradient scaling
            scaler.scale(loss).backward()
            
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        
        train_loss /= num_batches
        
        # Eval with amp too for speed
        with autocast():
            train_acc, _, *_ = model_eval(train_dataloader, model, device)
            dev_acc, _, *_ = model_eval(dev_dataloader, model, device)
        
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save({
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'args': None,
                'model_config': config_ns,
            }, filepath)
        
        print(f"  epoch {epoch}: loss={train_loss:.3f}, train_acc={train_acc:.3f}, dev_acc={dev_acc:.3f} (best={best_dev_acc:.3f})")
    
    # Load best model and eval on test
    saved = torch.load(filepath, weights_only=False)
    model.load_state_dict(saved['model'])
    model.eval()
    with autocast():
        dev_acc, _, dev_pred, _, dev_sents = model_eval(dev_dataloader, model, device)
    
    del model
    torch.cuda.empty_cache()
    
    return best_dev_acc, dev_acc


def run_experiment(name, ds, micro_bs, grad_accum, lr, epochs, dropout, device):
    """Run a single experiment."""
    seed_everything(1337)
    torch.cuda.empty_cache()
    
    tokenizer = Tokenizer(None)
    train_data, num_labels = create_data(ds["train"], tokenizer, 'train')
    dev_data = create_data(ds["dev"], tokenizer, 'valid')
    
    args_ns = SimpleNamespace(max_sentence_len=None, batch_size=micro_bs)
    train_dataset = LlamaDataset(train_data, args_ns)
    dev_dataset = LlamaDataset(dev_data, args_ns)
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=micro_bs,
                                  collate_fn=train_dataset.collate_fn, num_workers=0)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=micro_bs,
                                collate_fn=dev_dataset.collate_fn, num_workers=0)
    
    config_ns = SimpleNamespace(
        hidden_dropout_prob=dropout,
        pretrained_model_path='stories42M.pt',
        num_labels=num_labels,
        data_dir='.',
        option='finetune'
    )
    
    filepath = f"hp_search_{name.replace(' ', '_')}.pt"
    
    best_dev, final_dev = train_with_amp(
        config_ns, train_dataloader, dev_dataloader,
        lr, epochs, micro_bs, grad_accum, device, filepath
    )
    
    return best_dev, final_dev


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Using mixed precision (fp16) + gradient accumulation")
    print("=" * 90)
    
    results = []
    
    for i, (name, ds, micro_bs, grad_accum, lr, epochs, dropout) in enumerate(EXPERIMENTS):
        eff_bs = micro_bs * grad_accum
        print(f"\n[{i+1}/{len(EXPERIMENTS)}] {name}")
        print(f"  lr={lr}, epochs={epochs}, micro_bs={micro_bs}, grad_accum={grad_accum}, eff_bs={eff_bs}, dropout={dropout}")
        
        t0 = time.time()
        try:
            best_dev, final_dev = run_experiment(name, ds, micro_bs, grad_accum, lr, epochs, dropout, device)
            elapsed = time.time() - t0
            print(f"  >> Best Dev Acc: {best_dev:.4f} | Time: {elapsed:.0f}s")
            results.append({
                'name': name, 'lr': lr, 'epochs': epochs, 'eff_bs': eff_bs,
                'dropout': dropout, 'best_dev_acc': best_dev, 'time_s': elapsed
            })
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  >> OOM! Skipping. Try reducing micro_bs.")
                torch.cuda.empty_cache()
                results.append({
                    'name': name, 'lr': lr, 'epochs': epochs, 'eff_bs': eff_bs,
                    'dropout': dropout, 'best_dev_acc': -1, 'time_s': 0
                })
            else:
                raise
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n\n" + "=" * 90)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("=" * 90)
    print(f"{'Name':<25} {'LR':<10} {'Epochs':<7} {'BS':<5} {'Drop':<6} {'Dev Acc':<10} {'Time':<8}")
    print("-" * 90)
    
    sst_results = [r for r in results if 'SST' in r['name'] and r['best_dev_acc'] > 0]
    cfimdb_results = [r for r in results if 'CFIMDB' in r['name'] and r['best_dev_acc'] > 0]
    
    best_sst = max(sst_results, key=lambda x: x['best_dev_acc']) if sst_results else None
    best_cfimdb = max(cfimdb_results, key=lambda x: x['best_dev_acc']) if cfimdb_results else None
    
    for r in results:
        marker = ""
        if r == best_sst or r == best_cfimdb:
            marker = " <<< BEST"
        print(f"{r['name']:<25} {r['lr']:<10} {r['epochs']:<7} {r['eff_bs']:<5} {r['dropout']:<6} {r['best_dev_acc']:<10.4f} {r['time_s']:<8.0f}{marker}")
    
    # Save CSV
    with open("hyperparameter_results.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to hyperparameter_results.csv")
    if best_sst:
        print(f"\n>>> BEST SST-5:  {best_sst['name']} => Dev Acc = {best_sst['best_dev_acc']:.4f}")
    if best_cfimdb:
        print(f">>> BEST CFIMDB: {best_cfimdb['name']} => Dev Acc = {best_cfimdb['best_dev_acc']:.4f}")
