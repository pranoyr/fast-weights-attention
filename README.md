# Fast Weights Attention

Exploring fast-weight (linear-attention-style) architectures as alternatives/complements to
standard softmax self-attention.

## Recent developments

- **Linear Transformers Are Secretly Fast Weight Programmers** (Schlag, Irie & Schmidhuber, 2021)
  — reframes linear attention as a fast weight system and adds a **delta rule** update
  (write-then-erase instead of pure addition) to reduce interference between stored associations.
- **RetNet — Retentive Network** (Sun et al., 2023) — adds an exponential decay term to the
  recurrent state update, giving a "retention" mechanism with parallel (matmul), recurrent
  (O(1) per token), and chunkwise-recurrent forms.
- **RWKV** (Peng et al., 2023–2024, up through RWKV-6/7 "Goose"/"Finch") — an RNN with a
  linear-attention-like time-mixing block, channel-mixing, and learned per-channel decay, tuned
  to match transformer quality at LLM scale while keeping constant-memory inference.
- **Gated Linear Attention (GLA)** (Yang, Wang, Shen et al., 2024) — introduces data-dependent
  (input-conditioned) gating into the fast-weight recurrence and a hardware-efficient
  **chunkwise-parallel** training algorithm, closing much of the gap to softmax attention while
  staying subquadratic.
- **DeltaNet — Parallelizing Linear Transformers with the Delta Rule over Sequence Length**
  (Yang, Wang, Yu, Kim et al., 2024) — makes Schlag et al.'s delta-rule fast-weight update
  parallelizable across the sequence (via a chunkwise WY/UT-transform-style algorithm), so the
  more expressive erase-and-write update no longer forces sequential training.
- **Gated DeltaNet** (Yang, Kautz, Yin et al., 2024/2025, NVIDIA) — combines the delta rule with
  GLA-style gating in one recurrence, currently one of the strongest linear-recurrent
  architectures on language modeling benchmarks; adopted as a component of NVIDIA's
  Mamba2/attention hybrid line.
- **Mamba / Mamba-2** (Gu & Dao, 2023; Dao & Gu, 2024) — selective state-space models rather than
  literal fast-weight attention, but Mamba-2 proves a formal duality ("Structured State Space
  Duality") between SSM recurrences and (gated) linear attention, unifying the two lines of work.
- **Titans: Learning to Memorize at Test Time** (Behrouz, Zhong & Mirrokni, 2024, Google
  Research) — treats the fast-weight memory itself as a small neural network (an MLP) updated
  online by gradient descent at test time, with a "surprise"-based gating rule for what to write
  and forget, aimed at much longer effective context than fixed-size matrix states.
- **Test-Time Training (TTT) layers** (Sun, Li, Dalal et al., 2024) — generalizes the fast-weight
  idea further by making the hidden state a full learnable model updated via an explicit
  self-supervised gradient step per token, framing the RNN hidden state itself as "a model being
  trained at test time."
- **Chunkwise-parallel training as the common enabling trick** — across GLA, DeltaNet, Gated
  DeltaNet, and Mamba-2, the key systems contribution is a chunked, matmul-friendly form of the
  recurrence that trains at near-transformer throughput on GPUs/TPUs while keeping constant-size
  state and O(1)-per-token generation, closing the historical speed gap that made fast-weight
  models impractical at scale.

## Status

Project scaffold only — implementation has not started yet (see [main.py](main.py)).
