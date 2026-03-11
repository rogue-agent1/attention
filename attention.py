#!/usr/bin/env python3
"""Attention mechanisms — scaled dot-product, multi-head, causal, cross-attention.

Pure Python implementation of the core transformer building blocks.
No numpy/torch — just math and lists.

Usage: python attention.py [--test]
"""

import sys, math, random

def matmul(A, B):
    """Matrix multiply A (m×n) @ B (n×p) → (m×p)."""
    m, n, p = len(A), len(A[0]), len(B[0])
    C = [[0.0]*p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

def transpose(A):
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]

def softmax(x):
    mx = max(x)
    exps = [math.exp(v - mx) for v in x]
    s = sum(exps)
    return [e / s for e in exps]

def softmax_2d(X):
    return [softmax(row) for row in X]

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V.
    
    Q: (seq_q, d_k), K: (seq_k, d_k), V: (seq_k, d_v)
    Returns: (seq_q, d_v), attention_weights (seq_q, seq_k)
    """
    d_k = len(Q[0])
    scale = math.sqrt(d_k)
    
    # QK^T
    KT = transpose(K)
    scores = matmul(Q, KT)
    
    # Scale
    for i in range(len(scores)):
        for j in range(len(scores[0])):
            scores[i][j] /= scale
    
    # Apply mask (causal or padding)
    if mask is not None:
        for i in range(len(scores)):
            for j in range(len(scores[0])):
                if not mask[i][j]:
                    scores[i][j] = -1e9
    
    # Softmax
    weights = softmax_2d(scores)
    
    # Weighted sum of V
    output = matmul(weights, V)
    return output, weights

def linear_transform(X, W, b=None):
    """Apply linear transform: XW + b."""
    result = matmul(X, W)
    if b:
        for i in range(len(result)):
            for j in range(len(result[0])):
                result[i][j] += b[j]
    return result

def split_heads(X, n_heads):
    """Split last dim into n_heads: (seq, d) → list of n_heads × (seq, d/n_heads)."""
    seq_len = len(X)
    d = len(X[0])
    head_dim = d // n_heads
    heads = []
    for h in range(n_heads):
        head = [[X[s][h*head_dim + i] for i in range(head_dim)] for s in range(seq_len)]
        heads.append(head)
    return heads

def concat_heads(heads):
    """Concatenate heads: list of (seq, head_dim) → (seq, d)."""
    seq_len = len(heads[0])
    return [[v for head in heads for v in head[s]] for s in range(seq_len)]

def multi_head_attention(Q, K, V, n_heads, W_q, W_k, W_v, W_o, mask=None):
    """Multi-head attention.
    
    Q,K,V: (seq, d_model)
    W_q,W_k,W_v: (d_model, d_model), W_o: (d_model, d_model)
    """
    Q_proj = linear_transform(Q, W_q)
    K_proj = linear_transform(K, W_k)
    V_proj = linear_transform(V, W_v)
    
    Q_heads = split_heads(Q_proj, n_heads)
    K_heads = split_heads(K_proj, n_heads)
    V_heads = split_heads(V_proj, n_heads)
    
    head_outputs = []
    all_weights = []
    for qh, kh, vh in zip(Q_heads, K_heads, V_heads):
        out, weights = scaled_dot_product_attention(qh, kh, vh, mask)
        head_outputs.append(out)
        all_weights.append(weights)
    
    concat = concat_heads(head_outputs)
    output = linear_transform(concat, W_o)
    return output, all_weights

def causal_mask(seq_len):
    """Lower-triangular causal mask for autoregressive attention."""
    return [[1 if j <= i else 0 for j in range(seq_len)] for i in range(seq_len)]

def random_matrix(rows, cols, scale=0.1):
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

# --- Tests ---

def test_softmax():
    s = softmax([1, 2, 3])
    assert abs(sum(s) - 1.0) < 1e-9
    assert s[2] > s[1] > s[0]
    
    # Numerical stability
    s2 = softmax([1000, 1001, 1002])
    assert abs(sum(s2) - 1.0) < 1e-9

def test_matmul():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = matmul(A, B)
    assert C == [[19, 22], [43, 50]]

def test_scaled_attention():
    random.seed(42)
    seq_len, d = 4, 8
    Q = random_matrix(seq_len, d)
    K = random_matrix(seq_len, d)
    V = random_matrix(seq_len, d)
    
    out, weights = scaled_dot_product_attention(Q, K, V)
    assert len(out) == seq_len
    assert len(out[0]) == d
    # Weights should sum to 1 per row
    for row in weights:
        assert abs(sum(row) - 1.0) < 1e-6

def test_causal_attention():
    random.seed(42)
    seq_len, d = 4, 8
    Q = random_matrix(seq_len, d)
    K = random_matrix(seq_len, d)
    V = random_matrix(seq_len, d)
    mask = causal_mask(seq_len)
    
    _, weights = scaled_dot_product_attention(Q, K, V, mask)
    # Position 0 should only attend to itself
    assert weights[0][0] > 0.99
    # Position 1 attends to 0 and 1 only
    assert abs(weights[1][2]) < 1e-6
    assert abs(weights[1][3]) < 1e-6

def test_multi_head():
    random.seed(42)
    seq_len, d_model, n_heads = 3, 8, 2
    X = random_matrix(seq_len, d_model)
    W_q = random_matrix(d_model, d_model)
    W_k = random_matrix(d_model, d_model)
    W_v = random_matrix(d_model, d_model)
    W_o = random_matrix(d_model, d_model)
    
    out, weights = multi_head_attention(X, X, X, n_heads, W_q, W_k, W_v, W_o)
    assert len(out) == seq_len
    assert len(out[0]) == d_model
    assert len(weights) == n_heads

def test_cross_attention():
    random.seed(42)
    q_len, kv_len, d = 3, 5, 8
    Q = random_matrix(q_len, d)
    K = random_matrix(kv_len, d)
    V = random_matrix(kv_len, d)
    
    out, weights = scaled_dot_product_attention(Q, K, V)
    assert len(out) == q_len
    assert len(weights) == q_len
    assert len(weights[0]) == kv_len

def test_split_concat():
    X = [[1,2,3,4], [5,6,7,8]]
    heads = split_heads(X, 2)
    assert len(heads) == 2
    assert heads[0] == [[1,2],[5,6]]
    assert heads[1] == [[3,4],[7,8]]
    merged = concat_heads(heads)
    assert merged == X

if __name__ == "__main__":
    if "--test" in sys.argv or len(sys.argv) == 1:
        test_softmax()
        test_matmul()
        test_scaled_attention()
        test_causal_attention()
        test_multi_head()
        test_cross_attention()
        test_split_concat()
        print("All tests passed!")
