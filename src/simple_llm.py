#!/usr/bin/env python3
"""
最简LLM实现 - 用最少代码理解大语言模型核心原理
功能：实现一个包含注意力机制的简化版Transformer语言模型

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

# 设置随机种子，确保结果可复现
nt_seed(42)
random.seed(42)

class SimpleTokenizer:
    """简单的字符级分词器"""
    def __init__(self, text):
        # 获取所有唯一字符并排序，构建词汇表
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        # 字符到索引的映射
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        # 索引到字符的映射
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        """将文本编码为token索引列表"""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        """将token索引列表解码为文本"""
        return ''.join([self.idx_to_char[i] for i in indices])


class SimpleLLM(Module):
    """简化版大语言模型 — backed by notorch

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

        # Token嵌入层：将token索引转换为向量
        self.tok_emb = Embedding(vocab_size, d_model)

        # Transformer块堆叠
        self.layers = []
        for l in range(n_layers):
            layer = {
                'rms1': RMSNorm(d_model),          # 注意力前归一化
                'wq': Linear(d_model, d_model),      # Query投影
                'wk': Linear(d_model, d_model),      # Key投影
                'wv': Linear(d_model, d_model),      # Value投影
                'wo': Linear(d_model, d_model),      # 输出投影
                'rms2': RMSNorm(d_model),          # FFN前归一化
                'w_gate': Linear(d_model, hidden),   # SwiGLU gate
                'w_up': Linear(d_model, hidden),     # SwiGLU up
                'w_down': Linear(hidden, d_model),   # SwiGLU down
            }
            for k, v in layer.items():
                setattr(self, f'l{l}_{k}', v)
            self.layers.append(layer)

        # 最终归一化和输出头
        self.norm_f = RMSNorm(d_model)
        self.head = Linear(d_model, vocab_size)

    def param_list(self):
        """按前向传播顺序返回所有参数"""
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
        """前向传播（训练模式）通过 notorch tape。返回 (loss_idx, loss_val)。"""
        CTX = len(token_ids)
        DIM = self.d_model
        HD = self.head_dim

        _lib.nt_tape_start()
        _lib.nt_train_mode(1)

        params = self.param_list()
        tape_ids = [_lib.nt_tape_param(p._ptr) for p in params]
        _lib.nt_tape_no_decay(tape_ids[0])  # embedding不做权重衰减

        # 构建token和目标张量
        tok_t = Tensor.zeros(CTX)
        tgt_t = Tensor.zeros(CTX)
        tok_t.set_data([float(x) for x in token_ids])
        tgt_t.set_data([float(x) for x in target_ids])
        tok_idx = _lib.nt_tape_record(tok_t._ptr, 0, -1, -1, ctypes.c_float(0))
        tgt_idx = _lib.nt_tape_record(tgt_t._ptr, 0, -1, -1, ctypes.c_float(0))
        tok_t._owns = False
        tgt_t._owns = False

        # 前向: embedding → transformer blocks → rmsnorm → lm_head → cross_entropy
        pi = 0
        h = _lib.nt_seq_embedding(tape_ids[pi], -1, tok_idx, CTX, DIM); pi += 1

        for l in range(self.n_layers):
            rms1=tape_ids[pi]; pi+=1
            wq=tape_ids[pi]; pi+=1; wk=tape_ids[pi]; pi+=1
            wv=tape_ids[pi]; pi+=1; wo=tape_ids[pi]; pi+=1
            rms2=tape_ids[pi]; pi+=1
            wg=tape_ids[pi]; pi+=1; wu=tape_ids[pi]; pi+=1; wd=tape_ids[pi]; pi+=1

            # 多头注意力: RMSNorm → Q/K/V → RoPE → causal attention → 输出投影 + 残差
            xn = _lib.nt_seq_rmsnorm(h, rms1, CTX, DIM)
            q = _lib.nt_rope(_lib.nt_seq_linear(wq, xn, CTX), CTX, HD)
            k = _lib.nt_rope(_lib.nt_seq_linear(wk, xn, CTX), CTX, HD)
            v = _lib.nt_seq_linear(wv, xn, CTX)
            attn = _lib.nt_mh_causal_attention(q, k, v, CTX, HD)
            h = _lib.nt_add(h, _lib.nt_seq_linear(wo, attn, CTX))

            # SwiGLU前馈网络: RMSNorm → gate/up → SiLU(gate)*up → down + 残差
            xn = _lib.nt_seq_rmsnorm(h, rms2, CTX, DIM)
            gate = _lib.nt_silu(_lib.nt_seq_linear(wg, xn, CTX))
            up = _lib.nt_seq_linear(wu, xn, CTX)
            h = _lib.nt_add(h, _lib.nt_seq_linear(wd, _lib.nt_mul(gate, up), CTX))

        rmsf=tape_ids[pi]; pi+=1; head_i=tape_ids[pi]; pi+=1
        hf = _lib.nt_seq_rmsnorm(h, rmsf, CTX, DIM)
        logits_idx = _lib.nt_seq_linear(head_i, hf, CTX)
        loss_idx = _lib.nt_seq_cross_entropy(logits_idx, tgt_idx, CTX, self.vocab_size)

        # 从tape读取loss值
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
        """反向传播 + Chuck优化器 + 清除tape"""
        _lib.nt_tape_backward(loss_idx)
        _lib.nt_tape_clip_grads(ctypes.c_float(1.0))
        _lib.nt_tape_chuck_step(ctypes.c_float(lr), ctypes.c_float(loss_val))
        _lib.nt_tape_clear()

    def generate(self, tokenizer, prompt, max_new_tokens=50, temperature=0.8):
        """生成文本 — 自回归生成"""
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

            # 读取最后位置的logits
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

            # softmax采样
            probs = softmax(raw_logits)
            next_id = multinomial(probs)
            _lib.nt_tape_clear()
            ctx.append(next_id)

        return tokenizer.decode(ctx)


def train_simple_model():
    """训练简单模型的示例"""
    # 准备训练数据（这里使用一个简单的文本）
    text = """
    人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    机器学习是人工智能的一个重要分支，它通过算法使计算机能够从数据中学习并做出决策或预测。
    深度学习是机器学习的一个子集，它使用神经网络来模拟人脑的工作方式。
    大语言模型是深度学习在自然语言处理领域的重要应用，能够理解和生成人类语言。
    """

    # 初始化分词器和模型
    tokenizer = SimpleTokenizer(text)
    model = SimpleLLM(vocab_size=tokenizer.vocab_size)

    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"模型参数数量: {model.count_params():,}")

    # 准备训练数据
    input_ids = tokenizer.encode(text)
    lr = 0.001

    print("\n开始训练...")
    for epoch in range(100):
        # 随机选择一个序列片段
        start_idx = random.randint(0, max(0, len(input_ids) - model.max_seq_len - 1))
        end_idx = start_idx + model.max_seq_len

        # 输入和目标（目标是输入向右移动一位）
        token_ids = input_ids[start_idx:end_idx]
        target_ids = input_ids[start_idx+1:end_idx+1]

        # 前向传播 + 反向传播
        loss_idx, loss_val = model.forward_train(token_ids, target_ids)
        model.backward_step(loss_idx, loss_val, lr)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss_val:.4f}")

    print("训练完成！\n")

    # 测试生成
    print("=== 文本生成测试 ===")
    test_prompts = ["人工智能", "机器学习", "深度学习"]

    for prompt in test_prompts:
        generated_text = model.generate(tokenizer, prompt, max_new_tokens=30, temperature=0.8)
        print(f"输入: '{prompt}'")
        print(f"生成: {generated_text}")
        print("-" * 50)

if __name__ == "__main__":
    print("=== 最简LLM实现演示 ===")
    print("这是一个教学用的简化版大语言模型实现")
    print("包含了Transformer的核心组件：注意力机制、位置编码、前馈网络等\n")
    print("notorch version — no PyTorch. Pure C backend.\n")

    train_simple_model()
