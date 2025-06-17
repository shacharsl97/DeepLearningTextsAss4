from __future__ import annotations
import torch
import argparse
import os
from transformer import TransformerLM
import data
import lm
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None, help='Path to checkpoint directory to load model from')
    args = parser.parse_args()

    seq_len = 128
    batch_size = 64
    data_path = "data/"
    n_layers = 6
    n_heads = 6
    embed_size = 192
    mlp_hidden_size = embed_size * 4

    learning_rate = 5e-4
    gradient_clipping = 1.0

    num_batches_to_train = 50000

    tokenizer, tokenized_data = data.load_data(data_path)
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

    model: torch.nn.Module = TransformerLM(
            n_layers,
            n_heads,
            embed_size,
            seq_len,
            tokenizer.vocab_size(),
            mlp_hidden_size,
            with_residuals = True,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.95])

    if args.load is not None:
        # Load checkpoint
        checkpoint_dir = args.load
        # Find the latest checkpoint file in the directory
        ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
        latest_ckpt = sorted(ckpts)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_ckpt)
        tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer.json')
        model, optimizer, tokenizer, step, loss = lm.load_checkpoint(model, optimizer, checkpoint_path, tokenizer_path)
        print(f"Loaded checkpoint from {checkpoint_path} at step {step} with loss {loss}")
        model.eval()
        for i in range(10):
            prompt = "JULIET:"
            input_ids = tokenizer.tokenize(prompt)
            generated = model.sample_continuation(input_ids, 500)
            print(f"Generation {i+1}: '''{tokenizer.detokenize(generated)}'''")
        exit(0)

    start_time = time.time()
    model.train()
    num_batches = 0
    while True:
        for batch in data.batch_items(data_iter, batch_size):
            if num_batches >= num_batches_to_train: break
            num_batches = num_batches + 1

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)

            logits = model(batch_x)

            loss = lm.compute_loss(logits, batch_y, tokenizer.pad_id())

            # parameters update
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            num_batches += 1
            if num_batches % 10 == 0:
                print(f"Seen {num_batches} batches. last loss is: {loss.item()}")
                if num_batches % 100 == 0:
                    for _ in range(1):
                        model.eval()
                        sampled = tokenizer.detokenize(model.sample_continuation(tokenizer.tokenize("JULIET:"), 500))
                        model.train()
                        print(f"Model sample: '''{sampled}'''")
                    print("")
            if num_batches % 500 == 0:
                # Save checkpoint, including 3 generations and timestamp
                model.eval()
                generations = []
                for _ in range(3):
                    prompt = "JULIET:"
                    input_ids = tokenizer.tokenize(prompt)
                    generated = model.sample_continuation(input_ids, 500)
                    generations.append(tokenizer.detokenize(generated))
                model.train()
                hyperparams = {
                    'n_layers': n_layers,
                    'n_heads': n_heads,
                    'embed_size': embed_size,
                    'seq_len': seq_len,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'mlp_hidden_size': mlp_hidden_size,
                    'gradient_clipping': gradient_clipping,
                    'num_batches_to_train': num_batches_to_train,
                }
                elapsed = time.time() - start_time
                hyperparams['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                hyperparams['elapsed_seconds'] = elapsed
                print(f"Checkpoint at batch {num_batches}, elapsed {elapsed:.1f} seconds.")
                lm.save_checkpoint(model, optimizer, tokenizer, loss.item(), hyperparams, generations, checkpoint_dir='checkpoints', step=num_batches)
