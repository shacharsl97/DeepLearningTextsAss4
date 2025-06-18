from __future__ import annotations
import torch
import argparse
import os
from transformer import TransformerLM
import data
import lm
import time

DEVICE = torch.device('cpu')

def flip_rtl(text: str) -> str:
    # Flip the string for RTL display (simple reversal)
    return text[::-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None, help='Path to checkpoint directory to load model from')
    parser.add_argument('--hebrew', action='store_true', help='Use Hebrew data if set, otherwise use English')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number to use (default: 0)')
    parser.add_argument('--with_residuals', action='store_true', help='Use residual connections')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability (0.0 for no dropout)')
    parser.add_argument('--init', type=str, default='xavier', choices=['xavier', 'kaiming', 'normal'], help='Initialization method')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--embed_size', type=int, default=192, help='Embedding size')
    parser.add_argument('--mlp_hidden_size', type=int, default=None, help='MLP hidden size (default: 4*embed_size)')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')
    parser.add_argument('--search_run', action='store_true', help='If set, disables checkpoints and only samples at end (for run_search)')
    parser.add_argument('--num_batches_to_train', type=int, default=50000, help='Number of batches to train (default: 50000)')
    args = parser.parse_args()

    DEVICE = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    START_TOKEN = "בבוקר אחד" if args.hebrew else "JULIET:"

    seq_len = 128
    batch_size = 64
    data_path = "heb-data/" if args.hebrew else "data/"
    n_layers = args.n_layers
    n_heads = args.n_heads
    embed_size = args.embed_size
    mlp_hidden_size = args.mlp_hidden_size if args.mlp_hidden_size is not None else embed_size * 4

    learning_rate = args.learning_rate
    dropout = args.dropout
    with_residuals = args.with_residuals
    weight_decay = args.weight_decay

    num_batches_to_train = args.num_batches_to_train

    tokenizer, tokenized_data = data.load_data(data_path)
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

    model: torch.nn.Module = TransformerLM(
            n_layers,
            n_heads,
            embed_size,
            seq_len,
            tokenizer.vocab_size(),
            mlp_hidden_size,
            with_residuals=with_residuals,
            dropout=dropout,
            init_method=args.init,
        ).to(DEVICE)
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.95], weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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
            detok = tokenizer.detokenize(generated)
            print(f"Generation {i+1}: ", end="")
            if args.hebrew:
                print(f"'''{flip_rtl(detok)}'''")
            else:
                print(f"'''{detok}'''")
        exit(0)

    start_time = time.time()
    model.train()
    num_batches = 0
    search_run = args.search_run
    while num_batches < num_batches_to_train:
        for batch in data.batch_items(data_iter, batch_size):
            if num_batches >= num_batches_to_train:
                break

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits = model(batch_x)

            loss = lm.compute_loss(logits, batch_y, tokenizer.pad_id())

            # parameters update
            model.zero_grad()
            loss.backward()
            optimizer.step()

            num_batches += 1
            if search_run:
                if num_batches % 500 == 0:
                    print(f"Seen {num_batches} batches. last loss is: {loss.item()}")
            else:
                if num_batches % 10 == 0:
                    print(f"Seen {num_batches} batches. last loss is: {loss.item()}")
                    if num_batches % 500 == 0:
                        for _ in range(1):
                            model.eval()
                            sampled = tokenizer.detokenize(model.sample_continuation(tokenizer.tokenize(START_TOKEN), 500))
                            model.train()
                            print("Model sample: ", end="")
                            if args.hebrew:
                                print(f"'''{flip_rtl(sampled)}'''")
                            else:
                                print(f"'''{sampled}'''")
                        print("")
            if not search_run and num_batches % 500 == 0:
                # Save checkpoint, including 3 generations and timestamp
                model.eval()
                generations = []
                for _ in range(3):
                    prompt = "בבוקר אחד" if args.hebrew else "JULIET:"
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
                    'num_batches_to_train': num_batches_to_train,
                }
                elapsed = time.time() - start_time
                hyperparams['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                hyperparams['elapsed_seconds'] = elapsed
                lang = 'hebrew' if args.hebrew else 'english'
                print(f"Checkpoint at batch {num_batches}, elapsed {elapsed:.1f} seconds.")
                # Save each checkpoint in its own folder, including language
                checkpoint_dir = f'checkpoints/{lang}_step_{num_batches}'
                os.makedirs(checkpoint_dir, exist_ok=True)
                lm.save_checkpoint(model, optimizer, tokenizer, loss.item(), hyperparams, generations, checkpoint_dir=checkpoint_dir, step=num_batches)

    # At the end, if search_run, sample 3 generations only now
    if search_run:
        model.eval()
        for i in range(3):
            prompt = "בבוקר אחד" if args.hebrew else "JULIET:"
            input_ids = tokenizer.tokenize(prompt)
            generated = model.sample_continuation(input_ids, 500)
            detok = tokenizer.detokenize(generated)
            print("Model sample: ", end="")
            if args.hebrew:
                print(f"'''{flip_rtl(detok)}'''")
            else:
                print(f"'''{detok}'''")
        model.train()
