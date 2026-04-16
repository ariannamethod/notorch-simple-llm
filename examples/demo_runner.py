#!/usr/bin/env python3
"""
零基础LLM教程 - 非交互式演示代码
直接运行所有演示，无需用户输入

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

# 设置随机种子，确保结果可复现
nt_seed(42)
random.seed(42)

class SimpleTokenizer:
    """简单的字符级分词器"""
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
    """简化版大语言模型 — backed by notorch"""
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
    """演示分词器"""
    print("\n" + "="*60)
    print("🔤 分词器演示")
    print("="*60)

    sample_text = "人工智能很有趣"
    tokenizer = SimpleTokenizer(sample_text)

    print(f"📚 训练文本: '{sample_text}'")
    print(f"🔍 词汇表大小: {tokenizer.vocab_size}")
    print(f"📝 包含字符: {tokenizer.chars}")

    # 演示编码
    test_text = "人工智能"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"\n📝 编码演示:")
    print(f"   原始文本: '{test_text}'")
    print(f"   编码结果: {encoded}")
    print(f"   解码验证: '{decoded}'")

def demonstrate_model_training():
    """演示模型训练和生成"""
    print("\n" + "="*60)
    print("🎓 模型训练与生成演示")
    print("="*60)

    # 准备训练数据
    text = """人工智能是计算机科学的一个分支。机器学习是人工智能的重要组成部分。深度学习使用神经网络来模拟人脑。大语言模型能够理解和生成人类语言。自然语言处理是人工智能的重要应用领域。"""

    print(f"📚 训练数据长度: {len(text)} 字符")

    # 初始化分词器和模型
    tokenizer = SimpleTokenizer(text)
    model = SimpleLLM(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=4, n_layers=2)

    print(f"🤖 模型信息:")
    print(f"   词汇表大小: {tokenizer.vocab_size}")
    print(f"   模型参数: {model.count_params():,}")

    # 准备训练数据
    input_ids = tokenizer.encode(text)
    lr = 0.01

    print(f"\n🏋️ 开始训练...")

    for epoch in range(100):
        # 随机选择一个序列片段
        start_idx = random.randint(0, max(0, len(input_ids) - model.max_seq_len - 1))
        end_idx = start_idx + model.max_seq_len

        # 输入和目标
        token_ids = input_ids[start_idx:end_idx]
        target_ids = input_ids[start_idx+1:end_idx+1]

        # 前向传播 + 反向传播
        loss_idx, loss_val = model.forward_train(token_ids, target_ids)
        model.backward_step(loss_idx, loss_val, lr)

        if epoch % 25 == 0:
            print(f"   Epoch {epoch:3d}, Loss: {loss_val:.4f}")

    print(f"✅ 训练完成!")

    # 测试生成
    print(f"\n🎯 文本生成测试:")
    test_prompts = ["人工智能", "机器学习", "深度学习"]

    for prompt in test_prompts:
        print(f"\n📝 输入: '{prompt}'")
        generated_text = model.generate(tokenizer, prompt, max_new_tokens=15, temperature=0.8)
        print(f"🤖 生成: {generated_text}")

def demonstrate_attention_concept():
    """演示注意力机制概念"""
    print("\n" + "="*60)
    print("🔍 注意力机制概念演示")
    print("="*60)

    print("💡 注意力机制的生活例子:")
    print("   想象你在嘈杂的餐厅里和朋友聊天")
    print("   你的大脑会自动过滤掉周围的噪音")
    print("   专注于朋友的声音")
    print("   这就是注意力的作用!")

    print("\n🧠 在语言理解中:")
    print("   句子: '小明把书放在桌子上，然后他去了图书馆'")
    print("   当读到'他'时，注意力会回到'小明'")
    print("   因为我们知道'他'指的是小明")

    print("\n🔧 计算机如何实现:")
    print("   1. Query (查询): 我想了解什么？")
    print("   2. Key (键): 每个词能提供什么信息？")
    print("   3. Value (值): 每个词的具体内容")
    print("   4. 计算相关性分数")
    print("   5. 根据分数加权组合信息")

    print("\n🔧 notorch 实现:")
    print("   RMSNorm → Q/K/V投影 → RoPE位置编码 → 因果注意力 → 输出投影")
    print("   SwiGLU前馈: gate*up → down (比ReLU更高效)")

def main():
    """主函数 - 运行所有演示"""
    print("🚀 零基础LLM教程 - 完整演示")
    print("="*60)
    print("这个程序将展示LLM的核心组件和工作原理")
    print("notorch version — no PyTorch. Pure C backend.")

    # 运行所有演示
    demonstrate_tokenizer()
    demonstrate_attention_concept()
    demonstrate_model_training()

    print("\n" + "="*60)
    print("🎉 演示完成!")
    print("="*60)
    print("通过这些演示，你已经了解了:")
    print("✅ 分词器如何将文字转换为数字")
    print("✅ 注意力机制的基本概念")
    print("✅ 完整的LLM训练和生成过程")
    print("\n💡 下一步建议:")
    print("   1. 尝试修改模型参数，观察效果变化")
    print("   2. 使用更多的训练数据")
    print("   3. 学习更高级的LLM技术")
    print("   4. 阅读相关的研究论文")

if __name__ == "__main__":
    main()
