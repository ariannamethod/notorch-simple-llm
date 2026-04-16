#!/usr/bin/env python3
"""
Minimal LLM Implementation - Understand core LLM principles with minimal code
Function: Implement a simplified Transformer language model with attention mechanism

notorch version — no PyTorch. Pure C backend via ctypes.
RMSNorm, SwiGLU, RoPE, multi-head causal attention.
"""

import ctypes
import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ariannamethod.notorch_nn import (
    _lib, _get_tensor_struct, _NtTapeEntry, _NtTensor,
    Tensor, Parameter, Module, Linear, Embedding, RMSNorm,
    softmax, multinomial, seed as nt_seed,
)
from ariannamethod.chuck import ChuckOptimizer

# Set random seed to ensure reproducibility
nt_seed(42)
random.seed(42)

class SimpleTokenizer:
    """Simple character-level tokenizer"""
    def __init__(self, text):
        # Get all unique characters, sort them, and build the vocabulary
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        # Character to index mapping
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        # Index to character mapping
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        """Encode text into a list of token indices"""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        """Decode a list of token indices into text"""
        return ''.join([self.idx_to_char[i] for i in indices])


class SimpleLLM(Module):
    """Simplified large language model — backed by notorch

    Architecture: RMSNorm + SwiGLU + RoPE + multi-head causal attention.
    Same core as GPT/LLaMA, but tiny and educational.
    """
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, max_seq_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        # SwiGLU hidden dimension (2/3 * 4d, rounded to 64)
        hidden = int(d_model * 8 / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.hidden = hidden

        # Token embedding layer: convert token indices to vectors
        self.tok_emb = Embedding(vocab_size, d_model)

        # Stack of Transformer blocks
        self.layers = []
        for l in range(n_layers):
            layer = {
                'rms1': RMSNorm(d_model),          # Pre-attention normalization
                'wq': Linear(d_model, d_model),      # Query projection
                'wk': Linear(d_model, d_model),      # Key projection
                'wv': Linear(d_model, d_model),      # Value projection
                'wo': Linear(d_model, d_model),      # Output projection
                'rms2': RMSNorm(d_model),          # Pre-FFN normalization
                'w_gate': Linear(d_model, hidden),   # SwiGLU gate
                'w_up': Linear(d_model, hidden),     # SwiGLU up
                'w_down': Linear(hidden, d_model),   # SwiGLU down
            }
            for k, v in layer.items():
                setattr(self, f'l{l}_{k}', v)
            self.layers.append(layer)

        # Final normalization and output head
        self.norm_f = RMSNorm(d_model)
        self.head = Linear(d_model, vocab_size)

    def param_list(self):
        """Return all parameters in forward pass order"""
        params = [self.tok_emb.weight]
        for l in self.layers:
            params.extend([
                l['rms1'].weight, l['wq'].weight, l['wk'].weight,
                l['wv'].weight, l['wo'].weight, l['rms2'].weight,
                l['w_gate'].weight, l['w_up'].weight, l['w_down'].weight,
            ])
        params.extend([self.norm_f.weight, self.head.weight])
        return params

    def count_params(self):
        return sum(p.numel for p in self.param_list())

    def forward_train(self, token_ids, target_ids):
        """Forward pass (training mode) through notorch tape. Returns (loss_idx, loss_val)."""
        CTX = len(token_ids)
        DIM = self.d_model
        HD = self.head_dim

        _lib.nt_tape_start()
        _lib.nt_train_mode(1)

        params = self.param_list()
        tape_ids = [_lib.nt_tape_param(p._ptr) for p in params]
        _lib.nt_tape_no_decay(tape_ids[0])  # No weight decay for embedding

        # Build token and target tensors
        tok_t = Tensor.zeros(CTX)
        tgt_t = Tensor.zeros(CTX)
        tok_t.set_data([float(x) for x in token_ids])
        tgt_t.set_data([float(x) for x in target_ids])
        tok_idx = _lib.nt_tape_record(tok_t._ptr, 0, -1, -1, ctypes.c_float(0))
        tgt_idx = _lib.nt_tape_record(tgt_t._ptr, 0, -1, -1, ctypes.c_float(0))
        tok_t._owns = False
        tgt_t._owns = False

        # Forward: embedding → transformer blocks → rmsnorm → lm_head → cross_entropy
        pi = 0
        h = _lib.nt_seq_embedding(tape_ids[pi], -1, tok_idx, CTX, DIM); pi += 1

        for l in range(self.n_layers):
            rms1=tape_ids[pi]; pi+=1
            wq=tape_ids[pi]; pi+=1; wk=tape_ids[pi]; pi+=1
            wv=tape_ids[pi]; pi+=1; wo=tape_ids[pi]; pi+=1
            rms2=tape_ids[pi]; pi+=1
            wg=tape_ids[pi]; pi+=1; wu=tape_ids[pi]; pi+=1; wd=tape_ids[pi]; pi+=1

            # Multi-head attention: RMSNorm → Q/K/V → RoPE → causal attention → output projection + residual
            xn = _lib.nt_seq_rmsnorm(h, rms1, CTX, DIM)
            q = _lib.nt_rope(_lib.nt_seq_linear(wq, xn, CTX), CTX, HD)
            k = _lib.nt_rope(_lib.nt_seq_linear(wk, xn, CTX), CTX, HD)
            v = _lib.nt_seq_linear(wv, xn, CTX)
            attn = _lib.nt_mh_causal_attention(q, k, v, CTX, HD)
            h = _lib.nt_add(h, _lib.nt_seq_linear(wo, attn, CTX))

            # SwiGLU feed-forward network: RMSNorm → gate/up → SiLU(gate)*up → down + residual
            xn = _lib.nt_seq_rmsnorm(h, rms2, CTX, DIM)
            gate = _lib.nt_silu(_lib.nt_seq_linear(wg, xn, CTX))
            up = _lib.nt_seq_linear(wu, xn, CTX)
            h = _lib.nt_add(h, _lib.nt_seq_linear(wd, _lib.nt_mul(gate, up), CTX))

        rmsf=tape_ids[pi]; pi+=1; head_i=tape_ids[pi]; pi+=1
        hf = _lib.nt_seq_rmsnorm(h, rmsf, CTX, DIM)
        logits_idx = _lib.nt_seq_linear(head_i, hf, CTX)
        loss_idx = _lib.nt_seq_cross_entropy(logits_idx, tgt_idx, CTX, self.vocab_size)

        # Read loss value from tape
        tape_ptr = _lib.nt_tape_get()
        entry_size = ctypes.sizeof(_NtTapeEntry)
        tape_addr = ctypes.cast(tape_ptr, ctypes.c_void_p).value
        loss_entry = ctypes.cast(
            tape_addr + loss_idx * entry_size,
            ctypes.POINTER(_NtTapeEntry)
        ).contents
        loss_tensor = ctypes.cast(loss_entry.output, ctypes.POINTER(_NtTensor)).contents
        loss_val = loss_tensor.data[0]

        return loss_idx, loss_val

    def backward_step(self, loss_idx, loss_val, lr):
        """Backward pass + Chuck optimizer + clear tape"""
        _lib.nt_tape_backward(loss_idx)
        _lib.nt_tape_clip_grads(ctypes.c_float(1.0))
        _lib.nt_tape_chuck_step(ctypes.c_float(lr), ctypes.c_float(loss_val))
        _lib.nt_tape_clear()

    def generate(self, tokenizer, prompt, max_new_tokens=50, temperature=0.8):
        """Generate text — autoregressive generation"""
        _lib.nt_train_mode(0)
        ctx = tokenizer.encode(prompt)

        for _ in range(max_new_tokens):
            if len(ctx) > self.max_seq_len:
                ctx = ctx[-self.max_seq_len:]
            CTX = len(ctx)

            _lib.nt_tape_start()
            params = self.param_list()
            tape_ids = [_lib.nt_tape_param(p._ptr) for p in params]

            tok_t = Tensor.zeros(CTX)
            tgt_t = Tensor.zeros(CTX)
            tok_t.set_data([float(x) for x in ctx])
            tok_idx = _lib.nt_tape_record(tok_t._ptr, 0, -1, -1, ctypes.c_float(0))
            tgt_idx = _lib.nt_tape_record(tgt_t._ptr, 0, -1, -1, ctypes.c_float(0))
            tok_t._owns = False
            tgt_t._owns = False

            pi = 0
            h = _lib.nt_seq_embedding(tape_ids[pi], -1, tok_idx, CTX, self.d_model); pi += 1
            for l in range(self.n_layers):
                rms1=tape_ids[pi]; pi+=1
                wq=tape_ids[pi]; pi+=1; wk=tape_ids[pi]; pi+=1
                wv=tape_ids[pi]; pi+=1; wo=tape_ids[pi]; pi+=1
                rms2=tape_ids[pi]; pi+=1
                wg=tape_ids[pi]; pi+=1; wu=tape_ids[pi]; pi+=1; wd=tape_ids[pi]; pi+=1
                xn = _lib.nt_seq_rmsnorm(h, rms1, CTX, self.d_model)
                q = _lib.nt_rope(_lib.nt_seq_linear(wq, xn, CTX), CTX, self.head_dim)
                k = _lib.nt_rope(_lib.nt_seq_linear(wk, xn, CTX), CTX, self.head_dim)
                v = _lib.nt_seq_linear(wv, xn, CTX)
                attn = _lib.nt_mh_causal_attention(q, k, v, CTX, self.head_dim)
                h = _lib.nt_add(h, _lib.nt_seq_linear(wo, attn, CTX))
                xn = _lib.nt_seq_rmsnorm(h, rms2, CTX, self.d_model)
                gate = _lib.nt_silu(_lib.nt_seq_linear(wg, xn, CTX))
                up = _lib.nt_seq_linear(wu, xn, CTX)
                h = _lib.nt_add(h, _lib.nt_seq_linear(wd, _lib.nt_mul(gate, up), CTX))

            rmsf=tape_ids[pi]; pi+=1; head_i=tape_ids[pi]; pi+=1
            hf = _lib.nt_seq_rmsnorm(h, rmsf, CTX, self.d_model)
            logits_idx = _lib.nt_seq_linear(head_i, hf, CTX)

            # Read logits at the last position
            tape_ptr = _lib.nt_tape_get()
            entry_size = ctypes.sizeof(_NtTapeEntry)
            tape_addr = ctypes.cast(tape_ptr, ctypes.c_void_p).value
            logits_entry = ctypes.cast(
                tape_addr + logits_idx * entry_size,
                ctypes.POINTER(_NtTapeEntry)
            ).contents
            logits_t = ctypes.cast(logits_entry.output, ctypes.POINTER(_NtTensor)).contents
            offset = (CTX - 1) * self.vocab_size
            raw_logits = [logits_t.data[offset + i] / temperature for i in range(self.vocab_size)]

            # Softmax sampling
            probs = softmax(raw_logits)
            next_id = multinomial(probs)
            _lib.nt_tape_clear()
            ctx.append(next_id)

        return tokenizer.decode(ctx)


def train_simple_model():
    """Example of training a simple model"""
    # Prepare training data (using a simple text here)
    text = """
    Artificial intelligence is a branch of computer science that attempts to understand the essence of intelligence and produce intelligent machines that respond in ways similar to human intelligence.
    Machine learning is an important branch of artificial intelligence that uses algorithms to enable computers to learn from data and make decisions or predictions.
    Deep learning is a subset of machine learning that uses neural networks to simulate how the human brain works.
    Large language models are an important application of deep learning in natural language processing, capable of understanding and generating human language.
    """

    # Initialize tokenizer and model
    tokenizer = SimpleTokenizer(text)
    model = SimpleLLM(vocab_size=tokenizer.vocab_size)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model parameter count: {model.count_params():,}")

    # Prepare training data
    input_ids = tokenizer.encode(text)
    lr = 0.001

    print("\nStarting training...")
    for epoch in range(100):
        # Randomly select a sequence segment
        start_idx = random.randint(0, max(0, len(input_ids) - model.max_seq_len - 1))
        end_idx = start_idx + model.max_seq_len

        # Input and target (target is input shifted right by one position)
        token_ids = input_ids[start_idx:end_idx]
        target_ids = input_ids[start_idx+1:end_idx+1]

        # Forward pass + backward pass
        loss_idx, loss_val = model.forward_train(token_ids, target_ids)
        model.backward_step(loss_idx, loss_val, lr)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss_val:.4f}")

    print("Training complete!\n")

    # Test generation
    print("=== Text Generation Test ===")
    test_prompts = ["Artificial", "Machine le", "Deep learn"]

    for prompt in test_prompts:
        generated_text = model.generate(tokenizer, prompt, max_new_tokens=30, temperature=0.8)
        print(f"Input: '{prompt}'")
        print(f"Generated: {generated_text}")
        print("-" * 50)

if __name__ == "__main__":
    print("=== Minimal LLM Implementation Demo ===")
    print("This is an educational simplified large language model implementation")
    print("Contains core Transformer components: attention mechanism, positional encoding, feed-forward network, etc.\n")
    print("notorch version — no PyTorch. Pure C backend.\n")

    train_simple_model()
