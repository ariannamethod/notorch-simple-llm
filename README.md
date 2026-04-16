# Minimal LLM Implementation from Scratch

A clean, educational implementation of a Large Language Model (LLM) built from scratch using [notorch](https://github.com/ariannamethod/notorch) — a pure C neural network library. Zero Python dependencies. No PyTorch.

## 🎯 Purpose

This implementation is designed for **educational purposes** to help developers and researchers understand:
- How Transformer architecture works
- The mechanics of self-attention and multi-head attention
- Text generation with autoregressive models
- The building blocks of modern LLMs like GPT/LLaMA

📖 **[Read the Complete Tutorial](https://blog.csdn.net/jiaquan3011/article/details/149292522?fromshare=blogdetail&sharetype=blogdetail&sharerId=149292522&sharerefer=PC&sharesource=jiaquan3011&sharefrom=from_link)** - Comprehensive Chinese blog explaining LLM concepts and implementation details.

## 🚀 Features

- **Complete Transformer Implementation**: Multi-head attention, RoPE, SwiGLU, RMSNorm
- **Two Versions**: Detailed version with extensive comments and minimal version for core understanding
- **Zero Dependencies**: No PyTorch, no numpy — just a C compiler and Python
- **Lightweight**: Runs on CPU, no GPU required
- **Educational**: Clear code structure with detailed explanations
- **Runnable**: Works out of the box with minimal setup

## 📁 Files

- `src/simple_llm.py` - Full implementation with detailed comments
- `src/minimal_llm.py` - Streamlined version focusing on core logic
- `ariannamethod/` - notorch C library and Python bindings
- `examples/` - Demo scripts (interactive and non-interactive)

## 🛠️ Requirements

A C compiler. That's it.

```bash
# Build the notorch shared library
cd ariannamethod && cc -std=c11 -O2 -fPIC -shared -o libnotorch.so notorch.c -lm
```

No `pip install` needed. Zero dependencies.

## 🏃‍♂️ Quick Start

### Build notorch:
```bash
cd ariannamethod && cc -std=c11 -O2 -fPIC -shared -o libnotorch.so notorch.c -lm && cd ..
```

### Run the detailed version:
```bash
python src/simple_llm.py
```

### Run the minimal version:
```bash
python src/minimal_llm.py
```

## 📊 Model Architecture

```
SimpleLLM (notorch)
├── Token Embedding (vocab_size → d_model)
├── Transformer Blocks (n_layers)
│   ├── RMSNorm + Multi-Head Attention
│   │   ├── Query/Key/Value Linear Projections
│   │   ├── RoPE (Rotary Position Embedding)
│   │   ├── Scaled Dot-Product Attention
│   │   └── Causal Masking
│   ├── Residual Connection
│   ├── RMSNorm + SwiGLU Feed-Forward Network
│   │   ├── Gate Linear + SiLU
│   │   ├── Up Linear
│   │   └── Down Linear
│   └── Residual Connection
├── Final RMSNorm
└── Output Projection (d_model → vocab_size)
```

## 🔧 Key Components

### notorch — Pure C Neural Network Engine
The `ariannamethod/` directory contains:
- `notorch.c` / `notorch.h` — Complete neural network library in pure C
- `notorch_nn.py` — Python ctypes bindings (drop-in replacement for torch.nn)
- `chuck.py` — Chuck Optimizer (loss-aware, replaces Adam)

### Architecture (Modern LLM Design)
- **RMSNorm** instead of LayerNorm (more efficient)
- **SwiGLU** instead of ReLU FFN (better gradients)
- **RoPE** instead of learned positional embeddings (extrapolates to longer sequences)
- **Multi-head causal attention** with automatic masking

### Training via notorch Tape
```python
# Forward pass through C tape
loss_idx, loss_val = model.forward_train(token_ids, target_ids)
# Backward + Chuck optimizer + clear
model.backward_step(loss_idx, loss_val, lr)
```

### Text Generation
```python
result = model.generate(tokenizer, "人工智能", max_new_tokens=50)
```

## 📈 Example Output

```
=== 最简LLM实现演示 ===
词汇表大小: 86
模型参数数量: 448,640

开始训练...
Epoch 0, Loss: 4.9505
Epoch 20, Loss: 1.2039
Epoch 40, Loss: 0.3569
Epoch 60, Loss: 0.1626
Epoch 80, Loss: 0.1216
训练完成！

=== 文本生成测试 ===
输入: '人工智能'
生成: 人工智能的一个重要分支，它通过算法使计算机能够从数据中学习并做出决策
```

## 🎓 Educational Value

This implementation helps you understand:

1. **Attention Mechanism**: How models focus on relevant parts of input
2. **RoPE**: Rotary position embeddings for sequence order
3. **Autoregressive Generation**: How text is generated token by token
4. **Transformer Architecture**: The building blocks of modern LLMs
5. **Training Process**: Forward/backward through a computation tape

## 🔍 Code Walkthrough

### Simple Tokenizer
Character-level tokenization for educational purposes:
```python
class SimpleTokenizer:
    def encode(self, text): # text → token IDs
    def decode(self, indices): # token IDs → text
```

### Forward Pass (notorch tape)
```python
# Embedding lookup
h = _lib.nt_seq_embedding(...)

# Per layer: attention + SwiGLU
xn = _lib.nt_seq_rmsnorm(h, rms1, CTX, DIM)
q = _lib.nt_rope(_lib.nt_seq_linear(wq, xn, CTX), CTX, HD)
k = _lib.nt_rope(_lib.nt_seq_linear(wk, xn, CTX), CTX, HD)
v = _lib.nt_seq_linear(wv, xn, CTX)
attn = _lib.nt_mh_causal_attention(q, k, v, CTX, HD)
h = _lib.nt_add(h, _lib.nt_seq_linear(wo, attn, CTX))
```

## 📚 Learning Path

1. **📖 [Read the Complete Tutorial](https://blog.csdn.net/jiaquan3011/article/details/149292522?fromshare=blogdetail&sharetype=blogdetail&sharerId=149292522&sharerefer=PC&sharesource=jiaquan3011&sharefrom=from_link)** - Start with the comprehensive blog post (Chinese)
2. **Start with `src/minimal_llm.py`** - Understand the core structure
3. **Study `src/simple_llm.py`** - Learn detailed implementation
4. **Read the technical blog** - Understand the theory
5. **Experiment with parameters** - See how changes affect performance
6. **Extend the implementation** - Add your own improvements

## ⚡ Performance

- **Model Size**: ~400K parameters
- **Training Time**: < 1 minute on CPU
- **Memory Usage**: < 100MB
- **Dependencies**: 0 (just a C compiler)
- **Inference Speed**: Real-time text generation

## 🔬 Experiments to Try

1. **Change model size**: Increase `d_model`, `n_heads`, `n_layers`
2. **Modify training data**: Use different text datasets
3. **Adjust generation**: Try different temperature values
4. **Compare**: Run the same model with notorch vs PyTorch

## 🔗 References

- [notorch](https://github.com/ariannamethod/notorch) — Pure C neural network library
- [nanoGPT-notorch](https://github.com/ariannamethod/nanoGPT-notorch) — nanoGPT backed by notorch
- [llama2-notorch](https://github.com/ariannamethod/llama2-notorch) — LLaMA 2 backed by notorch

## 🤝 Contributing

This is an educational project. Feel free to:
- Report issues or bugs
- Suggest improvements
- Add more detailed explanations
- Create tutorials or examples

## 📄 License

MIT License - Feel free to use for educational purposes.

## 🙏 Acknowledgments

- Inspired by the "Attention Is All You Need" paper
- Educational approach influenced by Andrej Karpathy's tutorials
- Powered by [notorch](https://github.com/ariannamethod/notorch) — neural networks without PyTorch
- Built for the community to understand LLM fundamentals

## 📞 Contact

For questions about the implementation or suggestions for improvements, please open an issue.

---

**Note**: This is a simplified implementation for educational purposes. For production use, consider established frameworks or more optimized implementations.


