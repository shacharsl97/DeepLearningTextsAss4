from torch import nn
import torch
import torch.nn.functional as F
import attention
import mlp

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, mlp_hidden_size: int, max_context_len, with_residuals: bool = False, dropout: float = 0.0):
        super().__init__()
        self.causal_attention = attention.CausalSelfAttention(embed_size, n_heads, max_context_len)
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.with_residuals = with_residuals
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        if self.with_residuals:
            res = inputs
            x = self.layer_norm_1(res)
            x = self.causal_attention(x)
            x = self.dropout(x)
            x = x + res
            res = x
            x = self.layer_norm_2(res)
            x = self.mlp(x)
            x = x + res
            return x
        else:
            x = inputs
            x = self.layer_norm_1(x)
            x = self.causal_attention(x)
            x = self.dropout(x)  # Dropout after self-attention
            x = self.layer_norm_2(x)
            x = self.mlp(x)
            return x

class Embed(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_context_len, dropout: float = 0.0):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_context_len, embed_size)
        self.max_context_len = max_context_len
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len) of token indices
        batch_size, seq_len = x.size()
        # Token embeddings
        tok_embeddings = self.token_embeddings(x)  # (batch, seq_len, embed_size)
        # Position indices: shape (seq_len,) -> broadcast to (batch, seq_len)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embeddings = self.position_embeddings(positions)  # (batch, seq_len, embed_size)
        return self.dropout(tok_embeddings + pos_embeddings)  # Dropout after embedding


class TransformerLM(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
            dropout: float = 0.0,
            init_method: str = 'xavier',
            ):
        super().__init__()
        self.embed = Embed(vocab_size, embed_size, max_context_len, dropout=dropout)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(n_heads, embed_size, mlp_hidden_size, max_context_len, with_residuals, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.word_prediction = nn.Linear(embed_size, vocab_size)
        self.max_context_len = max_context_len
        self.init_weights(init_method)

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params/1e6,))

    def forward(self, inputs):
        x = self.embed(inputs)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        logits = self.word_prediction(x)
        return logits

    def init_weights(self, method='xavier'):
        for pn, p in self.named_parameters():
            if isinstance(p, nn.LayerNorm):
                torch.nn.init.zeros_(p.bias)
                torch.nn.init.ones_(p.weight)
            elif isinstance(p, nn.Linear):
                if method == 'xavier':
                    torch.nn.init.xavier_uniform_(p.weight)
                elif method == 'kaiming':
                    torch.nn.init.kaiming_uniform_(p.weight, nonlinearity='relu')
                elif method == 'normal':
                    torch.nn.init.normal_(p.weight, mean=0.0, std=0.02)
                if p.bias is not None:
                    torch.nn.init.zeros_(p.bias)
            elif isinstance(p, nn.Embedding):
                if method == 'xavier':
                    torch.nn.init.xavier_uniform_(p.weight)
                elif method == 'kaiming':
                    torch.nn.init.kaiming_uniform_(p.weight, nonlinearity='relu')
                elif method == 'normal':
                    torch.nn.init.normal_(p.weight, mean=0.0, std=0.02)


    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        device = next(self.parameters()).device  # Ensure input is on the same device as the model
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32, device=device))
                logits_for_last_token = logits[0][-1]
                distribution_for_last_token = F.softmax(logits_for_last_token, dim=-1)
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)
                generated.append(sampled_token.item())
                feed_to_lm.append(sampled_token.item())
        return generated

    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float, topK: int) -> list[int]:
        raise Exception("Not implemented")
        # TODO implement this.
        # Temperature should be the temperature in which you sample.
        # TopK indicates that we don't sample from the entire distribution, but only from the top k scoring tokens
        # for the given position.

