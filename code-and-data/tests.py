import torch
import attention
import torch.nn as nn

def test_attention_scores():
    # Simple test: batch=1, seq_a=2, seq_b=2, dim=2
    a = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # shape: (1, 2, 2)
    b = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # shape: (1, 2, 2)
    # Compute expected output manually:
    # For each pair (b_i, a_j): dot(b_i, a_j) / sqrt(2)
    # b_0 vs a_0: (1*1 + 0*0)/sqrt(2) = 1/sqrt(2)
    # b_0 vs a_1: (1*0 + 0*1)/sqrt(2) = 0
    # b_1 vs a_0: (0*1 + 1*0)/sqrt(2) = 0
    # b_1 vs a_1: (0*0 + 1*1)/sqrt(2) = 1/sqrt(2)
    sqrt2 = 2 ** 0.5
    expected_output = torch.tensor([[[1/sqrt2, 0.0], [0.0, 1/sqrt2]]])  # shape: (1, 2, 2)

    A = attention.attention_scores(a, b)
    assert torch.allclose(A, expected_output)

    # Additional, more complex test: batch=1, seq_a=3, seq_b=3, dim=2
    a = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # shape: (1, 3, 2)
    b = torch.tensor([[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])  # shape: (1, 3, 2)
    # Compute expected output manually:
    # Each score = dot(b_i, a_j) / sqrt(2)
    sqrt2 = 2 ** 0.5
    expected_output = torch.tensor([
        [
            [ (7*1+8*2)/sqrt2, (7*3+8*4)/sqrt2, (7*5+8*6)/sqrt2 ],
            [ (9*1+10*2)/sqrt2, (9*3+10*4)/sqrt2, (9*5+10*6)/sqrt2 ],
            [ (11*1+12*2)/sqrt2, (11*3+12*4)/sqrt2, (11*5+12*6)/sqrt2 ]
        ]
    ])  # shape: (1, 3, 3)
    A = attention.attention_scores(a, b)
    assert torch.allclose(A, expected_output)

def test_self_attention():
    # Simple test: batch=1, seq_len=2, dim=2
    v = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    # Attention scores: for simplicity, use zeros so softmax gives uniform weights
    A = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])  # (1, 2, 2)
    # Softmax over last dim: each row is [0.5, 0.5]
    # Output: each token gets 0.5*v[0] + 0.5*v[1]
    expected = torch.tensor([[[2.0, 3.0], [2.0, 3.0]]])  # (1, 2, 2)
    from attention import self_attention
    out = self_attention(v, A)
    assert torch.allclose(out, expected)

    # Test: non-uniform attention (one token dominates)
    v = torch.tensor([[[1.0, 2.0], [10.0, 20.0]]])  # (1, 2, 2)
    # Attention scores: make the second token much more likely
    A = torch.tensor([[[0.0, 5.0], [0.0, 5.0]]])  # (1, 2, 2)
    # Softmax([0, 5]) ~ [0.0067, 0.9933]
    w = torch.exp(torch.tensor([0.0, 5.0]))
    w = w / w.sum()
    expected = torch.stack([
        w[0]*v[0,0] + w[1]*v[0,1],
        w[0]*v[0,0] + w[1]*v[0,1]
    ]).unsqueeze(0)  # (1, 2, 2)
    out = self_attention(v, A)
    assert torch.allclose(out, expected, atol=1e-4)

def test_self_attention_causal():
    # Test causal masking: batch=1, seq_len=3, dim=1
    v = torch.tensor([[[1.0], [2.0], [4.0]]])  # (1, 3, 1)
    # Attention scores: all zeros (so softmax would be uniform if unmasked)
    A = torch.zeros((1, 3, 3))
    from attention import create_causal_mask, self_attention
    mask = create_causal_mask(embed_dim=1, n_heads=1, max_context_len=5)
    # With causal mask, for each i, only positions <= i are attended
    # For i=0: only v[0] (weight=1)
    # For i=1: v[0] and v[1] (weights=0.5, 0.5)
    # For i=2: v[0], v[1], v[2] (weights=1/3 each)
    expected = torch.tensor([[[1.0], [(1.0+2.0)/2], [(1.0+2.0+4.0)/3]]])
    out = self_attention(v, A, mask)
    assert torch.allclose(out, expected, atol=1e-5), f"\nExpected:\n{expected}\nGot:\n{out}"

def test_self_attention_layer():
    # Test self_attention_layer with batch=1, seq_len=2, dim=2
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # (1, 2, 2)
    from attention import create_kqv_matrix, create_causal_mask, self_attention_layer
    kqv_matrix = create_kqv_matrix(2)
    # Set weights to identity for deterministic output
    with torch.no_grad():
        kqv_matrix.weight.copy_(torch.eye(2).repeat(3, 1))
        kqv_matrix.bias.zero_()
    mask = create_causal_mask(embed_dim=2, n_heads=1, max_context_len=2)
    # Manually compute expected output: with causal mask, first token attends only to itself, second token attends to itself and first token
    # Compute attention scores for second token
    import math
    sqrt2 = math.sqrt(2)
    att_scores = torch.tensor([0.0, 1.0]) / sqrt2
    att_weights = torch.softmax(att_scores, dim=0)
    expected = torch.stack([
        torch.tensor([1.0, 0.0]),  # first token only attends to itself
        att_weights[0]*torch.tensor([1.0, 0.0]) + att_weights[1]*torch.tensor([0.0, 1.0])
    ]).unsqueeze(0)
    out = self_attention_layer(x, kqv_matrix, mask)
    assert torch.allclose(out, expected, atol=1e-5), f"\nExpected:\n{expected}\nGot:\n{out}"

def test_multi_head_attention_layer():
    # Test with 2 1s, batch=1, seq_len=2, dim=4 (so each 1 dim=2)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])  # (1, 2, 4)
    from attention import create_kqv_matrix, create_causal_mask, multi_head_attention_layer
    D = 4
    n_heads = 2
    D_1 = D // n_heads
    kqv_matrices = [create_kqv_matrix(D, n_heads) for _ in range(n_heads)]
    with torch.no_grad():
        # First 1: only use first 2 dims for k, q, v
        kqv_matrices[0].weight.zero_()
        kqv_matrices[0].bias.zero_()
        kqv_matrices[0].weight[:2, :2] = torch.eye(2)  # k
        kqv_matrices[0].weight[2:4, :2] = torch.eye(2)  # q
        kqv_matrices[0].weight[4:6, :2] = torch.eye(2)  # v
        # Second 1: only use last 2 dims for k, q, v
        kqv_matrices[1].weight.zero_()
        kqv_matrices[1].bias.zero_()
        kqv_matrices[1].weight[:2, 2:] = torch.eye(2)  # k
        kqv_matrices[1].weight[2:4, 2:] = torch.eye(2)  # q
        kqv_matrices[1].weight[4:6, 2:] = torch.eye(2)  # v
    mask = create_causal_mask(embed_dim=4, n_heads=2, max_context_len=2)
    # Run multi_head_attention_layer with full input x
    out = multi_head_attention_layer(x, kqv_matrices, mask)
    expected = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    if not torch.allclose(out, expected, atol=1e-5):
        print(f"Expected:\n{expected}\nGot:\n{out}")
    assert torch.allclose(out, expected, atol=1e-5), f"\nExpected:\n{expected}\nGot:\n{out}"

def test_causal_self_attention_module():
    # Test CausalSelfAttention with 2 1s, batch=1, seq_len=2, dim=4
    from attention import CausalSelfAttention
    torch.manual_seed(0)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])  # (1, 2, 4)
    model = CausalSelfAttention(embed_dim=4, n_heads=2, max_context_len=2)
    # Set kqv weights so each 1 just selects its own part of the input
    with torch.no_grad():
        model.kqv_matrices[0].weight.zero_()
        model.kqv_matrices[0].bias.zero_()
        model.kqv_matrices[0].weight[:2, :2] = torch.eye(2)  # k
        model.kqv_matrices[0].weight[2:4, :2] = torch.eye(2)  # q
        model.kqv_matrices[0].weight[4:6, :2] = torch.eye(2)  # v

        model.kqv_matrices[1].weight.zero_()
        model.kqv_matrices[1].bias.zero_()
        model.kqv_matrices[1].weight[:2, 2:] = torch.eye(2)  # k
        model.kqv_matrices[1].weight[2:4, 2:] = torch.eye(2)  # q
        model.kqv_matrices[1].weight[4:6, 2:] = torch.eye(2)  # v
        # Set proj to identity
        model.proj.weight.copy_(torch.eye(4))
        model.proj.bias.zero_()
    out = model(x)
    expected = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    assert torch.allclose(out, expected, atol=1e-5), f"\nExpected:\n{expected}\nGot:\n{out}"

if __name__ == "__main__":
    test_attention_scores()
    test_self_attention()
    test_self_attention_causal()
    test_self_attention_layer()
    test_multi_head_attention_layer()
    test_causal_self_attention_module()
    print("All tests passed!")