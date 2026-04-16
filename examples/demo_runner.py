#!/usr/bin/env python3
"""
Beginner LLM Tutorial - Non-interactive Demo Code
Runs all demos directly, no user input required

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

# Set random seed to ensure reproducible results
nt_seed(42)
random.seed(42)

class SimpleTokenizer:
    """Simple character-level tokenizer"""
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])


class SimpleLLM(Module):
    """Simplified large language model — backed by notorch"""
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

    def generate(self, tokenizer, prompt, max_new_tokens=50, temperature=0.8):
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

            probs = softmax(raw_logits)
            next_id = multinomial(probs)
            _lib.nt_tape_clear()
            ctx.append(next_id)

        return tokenizer.decode(ctx)


def demonstrate_tokenizer():
    """Demonstrate the tokenizer"""
    print("\n" + "="*60)
    print("🔤 Tokenizer Demo")
    print("="*60)

    sample_text = "AI is fascinating"
    tokenizer = SimpleTokenizer(sample_text)

    print(f"📚 Training text: '{sample_text}'")
    print(f"🔍 Vocabulary size: {tokenizer.vocab_size}")
    print(f"📝 Characters: {tokenizer.chars}")

    # Demonstrate encoding
    test_text = "AI is"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"\n📝 Encoding demo:")
    print(f"   Original text: '{test_text}'")
    print(f"   Encoded result: {encoded}")
    print(f"   Decoded verification: '{decoded}'")

def demonstrate_model_training():
    """Demonstrate model training and generation"""
    print("\n" + "="*60)
    print("🎓 Model Training and Generation Demo")
    print("="*60)

    # Prepare training data
    text = """Artificial intelligence is a branch of computer science. Machine learning is a key component of artificial intelligence. Deep learning uses neural networks to simulate the human brain. Large language models can understand and generate human language. Natural language processing is an important application area of artificial intelligence."""

    print(f"📚 Training data length: {len(text)} characters")

    # Initialize tokenizer and model
    tokenizer = SimpleTokenizer(text)
    model = SimpleLLM(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=4, n_layers=2)

    print(f"🤖 Model info:")
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    print(f"   Model parameters: {model.count_params():,}")

    # Prepare training data
    input_ids = tokenizer.encode(text)
    lr = 0.01

    print(f"\n🏋️ Starting training...")

    for epoch in range(100):
        # Randomly select a sequence segment
        start_idx = random.randint(0, max(0, len(input_ids) - model.max_seq_len - 1))
        end_idx = start_idx + model.max_seq_len

        # Input and target
        token_ids = input_ids[start_idx:end_idx]
        target_ids = input_ids[start_idx+1:end_idx+1]

        # Forward pass + backward pass
        loss_idx, loss_val = model.forward_train(token_ids, target_ids)
        model.backward_step(loss_idx, loss_val, lr)

        if epoch % 25 == 0:
            print(f"   Epoch {epoch:3d}, Loss: {loss_val:.4f}")

    print(f"✅ Training complete!")

    # Test generation
    print(f"\n🎯 Text generation test:")
    test_prompts = ["Artificial intel", "Machine learning", "Deep learning"]

    for prompt in test_prompts:
        print(f"\n📝 Input: '{prompt}'")
        generated_text = model.generate(tokenizer, prompt, max_new_tokens=15, temperature=0.8)
        print(f"🤖 Generated: {generated_text}")

def demonstrate_attention_concept():
    """Demonstrate the attention mechanism concept"""
    print("\n" + "="*60)
    print("🔍 Attention Mechanism Concept Demo")
    print("="*60)

    print("💡 A real-life example of attention:")
    print("   Imagine you are chatting with a friend in a noisy restaurant")
    print("   Your brain automatically filters out the surrounding noise")
    print("   and focuses on your friend's voice")
    print("   That is how attention works!")

    print("\n🧠 In language understanding:")
    print("   Sentence: 'Alice put the book on the table, then she went to the library'")
    print("   When reading 'she', attention goes back to 'Alice'")
    print("   Because we know 'she' refers to Alice")

    print("\n🔧 How computers implement this:")
    print("   1. Query: What do I want to know?")
    print("   2. Key: What information can each word provide?")
    print("   3. Value: The actual content of each word")
    print("   4. Compute relevance scores")
    print("   5. Combine information using weighted scores")

    print("\n🔧 notorch implementation:")
    print("   RMSNorm → Q/K/V projection → RoPE positional encoding → causal attention → output projection")
    print("   SwiGLU feed-forward: gate*up → down (more efficient than ReLU)")

def main():
    """Main function - run all demos"""
    print("🚀 Beginner LLM Tutorial - Full Demo")
    print("="*60)
    print("This program demonstrates the core components and workings of an LLM")
    print("notorch version — no PyTorch. Pure C backend.")

    # Run all demos
    demonstrate_tokenizer()
    demonstrate_attention_concept()
    demonstrate_model_training()

    print("\n" + "="*60)
    print("🎉 Demo complete!")
    print("="*60)
    print("Through these demos, you have learned about:")
    print("✅ How a tokenizer converts text into numbers")
    print("✅ The basic concept of the attention mechanism")
    print("✅ The complete LLM training and generation process")
    print("\n💡 Suggested next steps:")
    print("   1. Try modifying model parameters and observe the effects")
    print("   2. Use more training data")
    print("   3. Learn more advanced LLM techniques")
    print("   4. Read related research papers")

if __name__ == "__main__":
    main()
