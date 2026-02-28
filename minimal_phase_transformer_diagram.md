# Minimal Phase Transformer (Current No-Parameter Model)

This diagram explains `minimal_phase_transformer.py`.

## 1) Token Sequence Layout (LSD-first)

```text
index:   0    1    2    ...   10   11   12   13   ...   22   23
token: [BOS][X0][X1] ... [X9][C0][Y0][Y1] ... [Y10][EOS]

Xi = X[d1_i, d2_i] where i-th least-significant digits are paired.
Yi = Y[sum_digit_i, carry_out_i].
```

## 2) Autoregressive Step Logic

At generation step `k` (for digit `k`, LSD first):

```text
Inputs read:
- Pair token: Xk  (contains d1_k, d2_k)
- Carry token: previous output token (C0 for k=0, else Y_{k-1})

Compute:
s = d1_k + d2_k + carry_in

Phase map:
theta = s * (2*pi/10)
carry_out = floor(theta / (2*pi))    # equivalent to s >= 10
digit_out = round((theta mod 2*pi) / (2*pi/10)) mod 10

Emit:
Y[digit_out, carry_out]
```

For the final carry digit (`k=10`), it uses `d1=d2=0` and only propagates carry.
Then it emits `EOS`.

## 3) Dataflow Diagram

```text
                    +---------------------------+
X[d1,d2] ---------->| extract d1,d2             |
                    |                           |----+
prev token -------->| extract carry_in (0 or 1) |    |
(C0 or Y[.,c])      +---------------------------+    |
                                                     v
                                           +------------------+
                                           | s=d1+d2+carry_in |
                                           +------------------+
                                                     |
                                                     v
                                           +------------------+
                                           | phase transform  |
                                           | theta=s*(2*pi/10)|
                                           +------------------+
                                              |            |
                                              |            |
                                              v            v
                                  +----------------+  +----------------+
                                  | carry_out      |  | digit_out      |
                                  | floor(theta/2pi)| | theta mod 2pi   |
                                  +----------------+  +----------------+
                                              \            /
                                               \          /
                                                v        v
                                            +----------------+
                                            | emit Y[d,c]    |
                                            +----------------+
```

## 4) Why It Has ~No Parameters

- No learned attention matrices.
- No learned MLP.
- No learned embeddings.
- Only two numeric constants are stored: `2*pi` and `2*pi/10`.

So it is decoder-style in interface (`forward` + autoregressive generation), but computationally a fixed arithmetic program.
