#!/usr/bin/env python3
"""
LLM Tutorial from Scratch - Interactive Demo Code
This file contains all code from the tutorial and provides interactive demo features

notorch version — no PyTorch. Pure C backend via ctypes.
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

# Set random seed for reproducibility
nt_seed(42)
random.seed(42)

class SimpleTokenizer:
    """
    Simple character-level tokenizer

    This tokenizer converts text into numbers so the computer can process it
    Like assigning an ID number to each character
    """
    def __init__(self, text):
        print("🔤 Initializing tokenizer...")
        # Get all unique characters and sort them to build the vocabulary
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Character-to-index mapping (character → number)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        # Index-to-character mapping (number → character)
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        print(f"   Vocabulary size: {self.vocab_size}")
        print(f"   Characters: {self.chars[:10]}..." if len(self.chars) > 10 else f"   Characters: {self.chars}")

    def encode(self, text):
        """Encode text into a list of token indices"""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        """Decode a list of token indices into text"""
        return ''.join([self.idx_to_char[i] for i in indices])

    def demo_encoding(self, text):
        """Demonstrate the encoding process"""
        print(f"\n📝 Encoding demo:")
        print(f"   Original text: '{text}'")
        encoded = self.encode(text)
        print(f"   Encoded result: {encoded}")
        decoded = self.decode(encoded)
        print(f"   Decoded verification: '{decoded}'")
        return encoded


class SimpleLLM(Module):
    """
    Simplified large language model — backed by notorch

    This is our complete model, containing all core components of a modern LLM
    Although small, the principles are the same as GPT/LLaMA
    Architecture: RMSNorm + SwiGLU + RoPE + multi-head causal attention.
    """
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, max_seq_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        hidden = int(d_model * 8 / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.hidden = hidden

        print(f"\n🤖 Initializing SimpleLLM:")
        print(f"   Vocabulary size: {vocab_size}")
        print(f"   Model dimension: {d_model}")
        print(f"   Attention heads: {n_heads}")
        print(f"   Transformer layers: {n_layers}")
        print(f"   Max sequence length: {max_seq_len}")

        self.tok_emb = Embedding(vocab_size, d_model)
        self.layers = []
        for l in range(n_layers):
            layer = {
                'rms1': RMSNorm(d_model),
                'wq': Linear(d_model, d_model),
                'wk': Linear(d_model, d_model),
                'wv': Linear(d_model, d_model),
                'wo': Linear(d_model, d_model),
                'rms2': RMSNorm(d_model),
                'w_gate': Linear(d_model, hidden),
                'w_up': Linear(d_model, hidden),
                'w_down': Linear(hidden, d_model),
            }
            for k, v in layer.items():
                setattr(self, f'l{l}_{k}', v)
            self.layers.append(layer)
        self.norm_f = RMSNorm(d_model)
        self.head = Linear(d_model, vocab_size)

        print(f"   Total parameters: {self.count_params():,}")

    def param_list(self):
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
        CTX = len(token_ids)
        DIM = self.d_model
        HD = self.head_dim

        _lib.nt_tape_start()
        _lib.nt_train_mode(1)

        params = self.param_list()
        tape_ids = [_lib.nt_tape_param(p._ptr) for p in params]
        _lib.nt_tape_no_decay(tape_ids[0])

        tok_t = Tensor.zeros(CTX)
        tgt_t = Tensor.zeros(CTX)
        tok_t.set_data([float(x) for x in token_ids])
        tgt_t.set_data([float(x) for x in target_ids])
        tok_idx = _lib.nt_tape_record(tok_t._ptr, 0, -1, -1, ctypes.c_float(0))
        tgt_idx = _lib.nt_tape_record(tgt_t._ptr, 0, -1, -1, ctypes.c_float(0))
        tok_t._owns = False
        tgt_t._owns = False

        pi = 0
        h = _lib.nt_seq_embedding(tape_ids[pi], -1, tok_idx, CTX, DIM); pi += 1

        for l in range(self.n_layers):
            rms1=tape_ids[pi]; pi+=1
            wq=tape_ids[pi]; pi+=1; wk=tape_ids[pi]; pi+=1
            wv=tape_ids[pi]; pi+=1; wo=tape_ids[pi]; pi+=1
            rms2=tape_ids[pi]; pi+=1
            wg=tape_ids[pi]; pi+=1; wu=tape_ids[pi]; pi+=1; wd=tape_ids[pi]; pi+=1

            xn = _lib.nt_seq_rmsnorm(h, rms1, CTX, DIM)
            q = _lib.nt_rope(_lib.nt_seq_linear(wq, xn, CTX), CTX, HD)
            k = _lib.nt_rope(_lib.nt_seq_linear(wk, xn, CTX), CTX, HD)
            v = _lib.nt_seq_linear(wv, xn, CTX)
            attn = _lib.nt_mh_causal_attention(q, k, v, CTX, HD)
            h = _lib.nt_add(h, _lib.nt_seq_linear(wo, attn, CTX))

            xn = _lib.nt_seq_rmsnorm(h, rms2, CTX, DIM)
            gate = _lib.nt_silu(_lib.nt_seq_linear(wg, xn, CTX))
            up = _lib.nt_seq_linear(wu, xn, CTX)
            h = _lib.nt_add(h, _lib.nt_seq_linear(wd, _lib.nt_mul(gate, up), CTX))

        rmsf=tape_ids[pi]; pi+=1; head_i=tape_ids[pi]; pi+=1
        hf = _lib.nt_seq_rmsnorm(h, rmsf, CTX, DIM)
        logits_idx = _lib.nt_seq_linear(head_i, hf, CTX)
        loss_idx = _lib.nt_seq_cross_entropy(logits_idx, tgt_idx, CTX, self.vocab_size)

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
        _lib.nt_tape_backward(loss_idx)
        _lib.nt_tape_clip_grads(ctypes.c_float(1.0))
        _lib.nt_tape_chuck_step(ctypes.c_float(lr), ctypes.c_float(loss_val))
        _lib.nt_tape_clear()

    def generate(self, tokenizer, prompt, max_new_tokens=50, temperature=0.8, verbose=False):
        """
        Generate text

        Parameters:
        - prompt: input prompt
        - max_new_tokens: maximum number of tokens to generate
        - temperature: temperature parameter (controls randomness)
        - verbose: whether to show detailed process
        """
        _lib.nt_train_mode(0)
        ctx = tokenizer.encode(prompt)

        if verbose:
            print(f"\n🎯 Starting text generation:")
            print(f"   Input prompt: '{prompt}'")
            print(f"   Max generation length: {max_new_tokens}")
            print(f"   Temperature: {temperature}")
            print(f"   Encoded input: {ctx}")
            print(f"\n📝 Generation process:")

        for i in range(max_new_tokens):
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

            tape_ptr = _lib.nt_tape_get()
            entry_size = ctypes.sizeof(_NtTapeEntry)
            tape_addr = ctypes.cast(tape_ptr, ctypes.c_void_p).value
            logits_entry = ctypes.cast(
                tape_addr + logits_idx * entry_size,
                ctypes.POINTER(_NtTapeEntry)
            ).contents
            logits_t = ctypes.cast(logits_entry.output, ctypes.POINTER(_NtTensor)).contents
            offset = (CTX - 1) * self.vocab_size
            raw_logits = [logits_t.data[offset + j] / temperature for j in range(self.vocab_size)]

            probs = softmax(raw_logits)
            next_id = multinomial(probs)
            _lib.nt_tape_clear()
            ctx.append(next_id)

            if verbose and i < 10:
                next_char = tokenizer.decode([next_id])
                print(f"   Step {i+1}: generated '{next_char}' (token {next_id})")

        result = tokenizer.decode(ctx)
        if verbose:
            print(f"\n✅ Generation complete!")
            print(f"   Final result: '{result}'")
        return result


def demonstrate_tokenizer():
    """Demonstrate how the tokenizer works"""
    print("\n" + "="*60)
    print("🔤 Tokenizer Demo")
    print("="*60)

    sample_text = "AI is fascinating"
    tokenizer = SimpleTokenizer(sample_text)

    # Demonstrate encoding process
    tokenizer.demo_encoding("AI is")
    tokenizer.demo_encoding("fascinating")

    print(f"\n🔍 Vocabulary mapping:")
    for char, idx in list(tokenizer.char_to_idx.items())[:5]:
        print(f"   '{char}' → {idx}")

def demonstrate_attention():
    """Demonstrate how the attention mechanism works"""
    print("\n" + "="*60)
    print("🔍 Attention Mechanism Demo")
    print("="*60)

    print("\n💡 The attention mechanism is like eye movement when humans read:")
    print("   When we read a pronoun like 'he', our eyes look back to find who 'he' refers to")
    print("   The attention mechanism enables computers to do the same thing")

    print("\n🔧 Attention implementation in notorch:")
    print("   RMSNorm → Q/K/V projection → RoPE positional encoding → scaled dot-product attention → causal mask")
    print("   All computation is done at the C level, called via ctypes through libnotorch")

    # Demonstrate tensor operations
    t = Tensor.zeros(8)
    print(f"\n📊 Created notorch tensor: size {t.numel}")
    print("   notorch tensor operation successful!")

def train_and_demo():
    """Train the model and demonstrate text generation"""
    print("\n" + "="*60)
    print("🎓 Model Training and Generation Demo")
    print("="*60)

    # Prepare training data
    text = """Artificial intelligence is a branch of computer science. Machine learning is an important part of artificial intelligence. Deep learning uses neural networks to simulate the human brain. Large language models can understand and generate human language. Natural language processing is an important application area of artificial intelligence."""

    print(f"📚 Training data length: {len(text)} characters")

    # Initialize tokenizer and model
    tokenizer = SimpleTokenizer(text)
    model = SimpleLLM(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=4, n_layers=2)

    # Prepare training data
    input_ids = tokenizer.encode(text)
    lr = 0.01

    print(f"\n🏋️ Starting training...")
    losses = []

    for epoch in range(200):
        # Randomly select a sequence segment
        start_idx = random.randint(0, max(0, len(input_ids) - model.max_seq_len - 1))
        end_idx = start_idx + model.max_seq_len

        # Input and target
        token_ids = input_ids[start_idx:end_idx]
        target_ids = input_ids[start_idx+1:end_idx+1]

        # Forward pass + backward pass
        loss_idx, loss_val = model.forward_train(token_ids, target_ids)
        model.backward_step(loss_idx, loss_val, lr)

        losses.append(loss_val)

        if epoch % 50 == 0:
            print(f"   Epoch {epoch:3d}, Loss: {loss_val:.4f}")

    print(f"✅ Training complete! Final loss: {losses[-1]:.4f}")

    # Test generation
    print(f"\n🎯 Text generation test:")
    test_prompts = ["Artificial", "Machine", "Deep"]

    for prompt in test_prompts:
        print(f"\n📝 Input: '{prompt}'")
        generated_text = model.generate(tokenizer, prompt, max_new_tokens=20, temperature=0.8)
        print(f"🤖 Generated: {generated_text}")

def interactive_demo():
    """Interactive demo"""
    print("\n" + "="*60)
    print("🎮 Interactive Demo")
    print("="*60)

    print("Welcome to the LLM Interactive Demo! (notorch version)")
    print("You can choose from the following demos:")
    print("1. Tokenizer demo")
    print("2. Attention mechanism demo")
    print("3. Full training and generation demo")
    print("4. All demos")

    while True:
        try:
            choice = input("\nPlease choose (1-4, or 'q' to quit): ").strip()

            if choice == 'q':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                demonstrate_tokenizer()
            elif choice == '2':
                demonstrate_attention()
            elif choice == '3':
                train_and_demo()
            elif choice == '4':
                demonstrate_tokenizer()
                demonstrate_attention()
                train_and_demo()
            else:
                print("❌ Invalid choice, please enter 1-4 or 'q'")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    print("🚀 LLM Tutorial from Scratch - Interactive Demo")
    print("="*60)
    print("This program will walk you through the core components and principles of LLMs")
    print("notorch version — no PyTorch. Pure C backend.")

    # Run interactive demo
    interactive_demo()
