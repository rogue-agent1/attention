#!/usr/bin/env python3
"""Attention mechanism — scaled dot-product and multi-head attention.

One file. Zero deps. Does one thing well.

The core of transformers: Q·K^T/√d_k softmax then ·V, with multi-head
projection. Includes causal masking and positional encoding.
"""
import math, random, sys

def matmul(A, B):
    """Matrix multiply: (m,n) x (n,p) -> (m,p)."""
    m, n = len(A), len(A[0])
    p = len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)] for i in range(m)]

def transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def softmax(row):
    mx = max(row)
    exps = [math.exp(x - mx) for x in row]
    s = sum(exps)
    return [e / s for e in exps]

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (seq_q, d_k), K: (seq_k, d_k), V: (seq_k, d_v)
    Returns: (seq_q, d_v), attention_weights
    """
    d_k = len(Q[0])
    scale = math.sqrt(d_k)
    # Q · K^T
    KT = transpose(K)
    scores = matmul(Q, KT)
    # Scale
    scores = [[s / scale for s in row] for row in scores]
    # Mask (causal or padding)
    if mask is not None:
        for i in range(len(scores)):
            for j in range(len(scores[0])):
                if not mask[i][j]:
                    scores[i][j] = -1e9
    # Softmax per row
    weights = [softmax(row) for row in scores]
    # Weights · V
    output = matmul(weights, V)
    return output, weights

def linear_project(X, W, b=None):
    """X: (seq, d_in), W: (d_in, d_out) -> (seq, d_out)."""
    result = matmul(X, W)
    if b:
        result = [[result[i][j] + b[j] for j in range(len(b))] for i in range(len(result))]
    return result

def random_matrix(rows, cols, scale=0.1):
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Projection matrices
        self.W_q = random_matrix(d_model, d_model)
        self.W_k = random_matrix(d_model, d_model)
        self.W_v = random_matrix(d_model, d_model)
        self.W_o = random_matrix(d_model, d_model)

    def _split_heads(self, X):
        """(seq, d_model) -> (num_heads, seq, d_k)."""
        seq_len = len(X)
        heads = []
        for h in range(self.num_heads):
            start = h * self.d_k
            head = [[X[s][start + k] for k in range(self.d_k)] for s in range(seq_len)]
            heads.append(head)
        return heads

    def _merge_heads(self, heads):
        """(num_heads, seq, d_k) -> (seq, d_model)."""
        seq_len = len(heads[0])
        merged = []
        for s in range(seq_len):
            row = []
            for h in heads:
                row.extend(h[s])
            merged.append(row)
        return merged

    def forward(self, X, causal=False):
        Q = linear_project(X, self.W_q)
        K = linear_project(X, self.W_k)
        V = linear_project(X, self.W_v)
        Q_heads = self._split_heads(Q)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)
        # Causal mask
        mask = None
        if causal:
            seq = len(X)
            mask = [[1 if j <= i else 0 for j in range(seq)] for i in range(seq)]
        attn_heads = []
        all_weights = []
        for qh, kh, vh in zip(Q_heads, K_heads, V_heads):
            out, w = scaled_dot_product_attention(qh, kh, vh, mask)
            attn_heads.append(out)
            all_weights.append(w)
        merged = self._merge_heads(attn_heads)
        output = linear_project(merged, self.W_o)
        return output, all_weights

def positional_encoding(seq_len, d_model):
    pe = []
    for pos in range(seq_len):
        row = []
        for i in range(d_model):
            if i % 2 == 0:
                row.append(math.sin(pos / (10000 ** (i / d_model))))
            else:
                row.append(math.cos(pos / (10000 ** ((i - 1) / d_model))))
        pe.append(row)
    return pe

def main():
    random.seed(42)
    print("=== Attention Mechanism ===\n")

    # Simple attention
    seq_len, d_model = 4, 8
    X = random_matrix(seq_len, d_model)
    Q = K = V = X
    out, weights = scaled_dot_product_attention(Q, K, V)
    print(f"Self-attention: input ({seq_len}, {d_model}) → output ({len(out)}, {len(out[0])})")
    print(f"Attention weights (row sums ≈ 1.0): {[f'{sum(w):.3f}' for w in weights]}")

    # Causal attention
    mask = [[1 if j <= i else 0 for j in range(seq_len)] for i in range(seq_len)]
    out_c, weights_c = scaled_dot_product_attention(Q, K, V, mask)
    print(f"\nCausal mask pattern:")
    for row in mask:
        print(f"  {''.join('█' if x else '░' for x in row)}")

    # Multi-head attention
    print(f"\nMulti-Head Attention (d_model={d_model}, heads=2):")
    mha = MultiHeadAttention(d_model, num_heads=2)
    out_mha, head_weights = mha.forward(X, causal=True)
    print(f"  Output shape: ({len(out_mha)}, {len(out_mha[0])})")
    print(f"  Heads: {len(head_weights)}, each with {len(head_weights[0])}x{len(head_weights[0][0])} weights")

    # Positional encoding
    pe = positional_encoding(seq_len, d_model)
    X_pe = [[X[i][j] + pe[i][j] for j in range(d_model)] for i in range(seq_len)]
    print(f"\nWith positional encoding: input[0][:4] = {[f'{x:.3f}' for x in X_pe[0][:4]]}")

if __name__ == "__main__":
    main()
