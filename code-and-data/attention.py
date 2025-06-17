from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import math


def create_kqv_matrix(input_vector_dim, n_heads = 1):
    # Projects from input_vector_dim (full embedding) to 3*output_head_dim (k, q, v for this head)
    assert input_vector_dim % n_heads == 0, f"input_vector_dim {input_vector_dim} must be divisible by n_heads {n_heads}"
    output_head_dim = input_vector_dim // n_heads
    return nn.Linear(input_vector_dim, 3 * output_head_dim)

def kqv(x, linear):
    # x: (B, N, D)
    kqv_out = linear(x)  # (B, N, 3*D)
    k, q, v = torch.chunk(kqv_out, 3, dim=-1)  # Each is (B, N, D)
    return k, q, v

def attention_scores(a, b):
    B1, N1, D1 = a.size()
    B2, N2, D2 = b.size()
    assert B1 == B2
    assert D1 == D2

    A = torch.bmm(b, a.transpose(1, 2))  # (B, N2, N1)
    A = A / math.sqrt(D1)  # Scale by sqrt(D1)
    # A is now (B, N2, N1) where A[b_i, a_j] = dot(b_i, a_j) / sqrt(D1)
    return A

def create_causal_mask(embed_dim, n_heads, max_context_len):
    # Create a mask of shape (1, max_context_len, max_context_len)
    # mask[i, j] = 1 if j <= i else 0
    mask = torch.tril(torch.ones((max_context_len, max_context_len), dtype=torch.bool))
    mask = mask.unsqueeze(0)  # (1, max_context_len, max_context_len)
    return mask

def self_attention(v, A, mask = None):
    # A: (B, N, N), v: (B, N, D)
    if mask is not None:
        # mask: (1, max_context_len, max_context_len)
        N = A.size(1)
        # Slice mask to current sequence length
        mask_slice = mask[:, :N, :N]  # (1, N, N)
        # Set masked positions to -inf
        A = A.masked_fill(~mask_slice, float('-inf'))
    attn_weights = F.softmax(A, dim=-1)  # (B, N, N)
    # Weighted sum of value vectors
    sa = torch.bmm(attn_weights, v)  # (B, N, D)
    return sa


def self_attention_layer(x, kqv_matrix, attention_mask):
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(k, q)
    sa = self_attention(v, att, attention_mask)
    return sa

def multi_head_attention_layer(x, kqv_matrices, mask):
    # x: (B, N, D), kqv_matrices: list of nn.Linear, mask: (1, max_context_len, max_context_len)
    B, N, D = x.size()
    n_heads = len(kqv_matrices)
    assert D % n_heads == 0, f"Input dimension D={D} must be a multiple of n_heads={n_heads}"
    D_head = D // n_heads
    for kqv_matrix in kqv_matrices:
        assert kqv_matrix.in_features == D, f"Each kqv_matrix must take input D={D}, got {kqv_matrix.in_features}"
        assert kqv_matrix.out_features == 3 * D_head, f"Each kqv_matrix must output 3*D_head={3*D_head}, got {kqv_matrix.out_features}"
    head_outputs = []
    for kqv_matrix in kqv_matrices:
        head_out = self_attention_layer(x, kqv_matrix, mask)  # (B, N, D_head)
        head_outputs.append(head_out)
    # Concatenate along the last dimension
    sa = torch.cat(head_outputs, dim=-1)  # (B, N, D)
    assert sa.size() == x.size(), f"\nSA:\n{sa.size()}\nX:\n{x.size()}"
    return sa


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len):
        super().__init__()
        assert embed_dim % n_heads == 0
        D_head = embed_dim // n_heads
        # Each head gets its own kqv matrix for D_head, projecting from embed_dim
        self.kqv_matrices = nn.ModuleList([create_kqv_matrix(embed_dim, n_heads) for _ in range(n_heads)])
        mask = create_causal_mask(embed_dim, n_heads, max_context_len)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        sa = multi_head_attention_layer(x, self.kqv_matrices, self.mask)
        sa = self.proj(sa)
        return sa
