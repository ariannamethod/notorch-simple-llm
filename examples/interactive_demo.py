#!/usr/bin/env python3
"""
零基础LLM教程 - 交互式演示代码
这个文件包含了教程中所有的代码，并提供了交互式的演示功能

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
    """
    简单的字符级分词器

    这个分词器把文本转换成数字，让计算机能够处理
    就像给每个字符分配一个身份证号码
    """
    def __init__(self, text):
        print("🔤 初始化分词器...")
        # 获取所有唯一字符并排序，构建词汇表
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # 字符到索引的映射（字符 → 数字）
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        # 索引到字符的映射（数字 → 字符）
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        print(f"   词汇表大小: {self.vocab_size}")
        print(f"   包含字符: {self.chars[:10]}..." if len(self.chars) > 10 else f"   包含字符: {self.chars}")

    def encode(self, text):
        """将文本编码为token索引列表"""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        """将token索引列表解码为文本"""
        return ''.join([self.idx_to_char[i] for i in indices])

    def demo_encoding(self, text):
        """演示编码过程"""
        print(f"\n📝 编码演示:")
        print(f"   原始文本: '{text}'")
        encoded = self.encode(text)
        print(f"   编码结果: {encoded}")
        decoded = self.decode(encoded)
        print(f"   解码验证: '{decoded}'")
        return encoded


class SimpleLLM(Module):
    """
    简化版大语言模型 — backed by notorch

    这是我们的完整模型，包含了现代LLM的所有核心组件
    虽然很小，但原理和GPT/LLaMA是一样的
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

        print(f"\n🤖 初始化SimpleLLM:")
        print(f"   词汇表大小: {vocab_size}")
        print(f"   模型维度: {d_model}")
        print(f"   注意力头数: {n_heads}")
        print(f"   Transformer层数: {n_layers}")
        print(f"   最大序列长度: {max_seq_len}")

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

        print(f"   总参数数量: {self.count_params():,}")

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
        生成文本

        参数:
        - prompt: 输入提示
        - max_new_tokens: 最大生成token数
        - temperature: 温度参数（控制随机性）
        - verbose: 是否显示详细过程
        """
        _lib.nt_train_mode(0)
        ctx = tokenizer.encode(prompt)

        if verbose:
            print(f"\n🎯 开始生成文本:")
            print(f"   输入提示: '{prompt}'")
            print(f"   最大生成长度: {max_new_tokens}")
            print(f"   温度参数: {temperature}")
            print(f"   编码后的输入: {ctx}")
            print(f"\n📝 生成过程:")

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
                print(f"   步骤 {i+1}: 生成 '{next_char}' (token {next_id})")

        result = tokenizer.decode(ctx)
        if verbose:
            print(f"\n✅ 生成完成!")
            print(f"   最终结果: '{result}'")
        return result


def demonstrate_tokenizer():
    """演示分词器的工作原理"""
    print("\n" + "="*60)
    print("🔤 分词器演示")
    print("="*60)

    sample_text = "人工智能很有趣"
    tokenizer = SimpleTokenizer(sample_text)

    # 演示编码过程
    tokenizer.demo_encoding("人工智能")
    tokenizer.demo_encoding("很有趣")

    print(f"\n🔍 词汇表映射:")
    for char, idx in list(tokenizer.char_to_idx.items())[:5]:
        print(f"   '{char}' → {idx}")

def demonstrate_attention():
    """演示注意力机制的工作原理"""
    print("\n" + "="*60)
    print("🔍 注意力机制演示")
    print("="*60)

    print("\n💡 注意力机制就像人类阅读时的眼球运动:")
    print("   当我们读到'他'这个代词时，眼睛会回头寻找'他'指的是谁")
    print("   注意力机制让计算机也能做到这一点")

    print("\n🔧 notorch 中的注意力实现:")
    print("   RMSNorm → Q/K/V投影 → RoPE位置编码 → 缩放点积注意力 → 因果掩码")
    print("   所有计算在C层面完成，通过ctypes调用libnotorch")

    # 演示张量操作
    t = Tensor.zeros(8)
    print(f"\n📊 创建notorch张量: 大小 {t.numel}")
    print("   notorch张量操作成功!")

def train_and_demo():
    """训练模型并演示生成效果"""
    print("\n" + "="*60)
    print("🎓 模型训练与生成演示")
    print("="*60)

    # 准备训练数据
    text = """人工智能是计算机科学的一个分支。机器学习是人工智能的重要组成部分。深度学习使用神经网络来模拟人脑。大语言模型能够理解和生成人类语言。自然语言处理是人工智能的重要应用领域。"""

    print(f"📚 训练数据长度: {len(text)} 字符")

    # 初始化分词器和模型
    tokenizer = SimpleTokenizer(text)
    model = SimpleLLM(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=4, n_layers=2)

    # 准备训练数据
    input_ids = tokenizer.encode(text)
    lr = 0.01

    print(f"\n🏋️ 开始训练...")
    losses = []

    for epoch in range(200):
        # 随机选择一个序列片段
        start_idx = random.randint(0, max(0, len(input_ids) - model.max_seq_len - 1))
        end_idx = start_idx + model.max_seq_len

        # 输入和目标
        token_ids = input_ids[start_idx:end_idx]
        target_ids = input_ids[start_idx+1:end_idx+1]

        # 前向传播 + 反向传播
        loss_idx, loss_val = model.forward_train(token_ids, target_ids)
        model.backward_step(loss_idx, loss_val, lr)

        losses.append(loss_val)

        if epoch % 50 == 0:
            print(f"   Epoch {epoch:3d}, Loss: {loss_val:.4f}")

    print(f"✅ 训练完成! 最终损失: {losses[-1]:.4f}")

    # 测试生成
    print(f"\n🎯 文本生成测试:")
    test_prompts = ["人工智能", "机器学习", "深度学习"]

    for prompt in test_prompts:
        print(f"\n📝 输入: '{prompt}'")
        generated_text = model.generate(tokenizer, prompt, max_new_tokens=20, temperature=0.8)
        print(f"🤖 生成: {generated_text}")

def interactive_demo():
    """交互式演示"""
    print("\n" + "="*60)
    print("🎮 交互式演示")
    print("="*60)

    print("欢迎来到LLM交互式演示! (notorch版)")
    print("你可以选择以下演示:")
    print("1. 分词器演示")
    print("2. 注意力机制演示")
    print("3. 完整训练和生成演示")
    print("4. 全部演示")

    while True:
        try:
            choice = input("\n请选择 (1-4, 或 'q' 退出): ").strip()

            if choice == 'q':
                print("👋 再见!")
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
                print("❌ 无效选择，请输入 1-4 或 'q'")

        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    print("🚀 零基础LLM教程 - 交互式演示")
    print("="*60)
    print("这个程序将带你体验LLM的核心组件和工作原理")
    print("notorch version — no PyTorch. Pure C backend.")

    # 运行交互式演示
    interactive_demo()
