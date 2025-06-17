from __future__ import annotations
import os
import json
import torch
from torch import nn
import torch.nn.functional as F

def batch_to_labeled_samples(batch: torch.IntTensor) -> tuple[torch.IntTensor, torch.IntTensor]:
    # batch: (batch_size, seq_len+1)
    # Inputs: all but last character; Labels: all but first character
    inputs = batch[:, :-1]
    labels = batch[:, 1:]
    return inputs, labels

def compute_loss(logits, gold_labels, pad_id: int = None):
    # logits: (batch, seq_len, vocab_size)
    # gold_labels: (batch, seq_len)
    # If pad_id is given, ignore padding tokens in the loss
    # PyTorch's cross_entropy expects (batch*seq_len, vocab_size) and (batch*seq_len,)
    batch, seq_len, vocab_size = logits.size()
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = gold_labels.reshape(-1)
    if pad_id is not None:
        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=pad_id)
    else:
        loss = F.cross_entropy(logits_flat, labels_flat)
    return loss

def save_checkpoint(model, optimizer, tokenizer, loss, hyperparams, generations, checkpoint_dir, step):
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Save model and optimizer state
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step{step}_loss{loss:.4f}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'loss': loss
    }, checkpoint_path)
    # Save tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    # Save hyperparams and generations
    props = dict(hyperparams)
    props['loss'] = loss
    props['step'] = step
    props['generations'] = generations
    props_path = os.path.join(checkpoint_dir, f"checkpoint_step{step}_props.json")
    with open(props_path, 'w', encoding='utf-8') as f:
        json.dump(props, f, ensure_ascii=False, indent=2)

def load_checkpoint(model, optimizer, checkpoint_path, tokenizer_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', None)
    from data import CharTokenizer
    tokenizer = CharTokenizer.load(tokenizer_path)
    return model, optimizer, tokenizer, step, loss

