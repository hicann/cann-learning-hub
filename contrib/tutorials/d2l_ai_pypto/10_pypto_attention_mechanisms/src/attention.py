"""第 10 章（注意力机制）共享组件模块。

与 d2l 原书 ``#@save``（首次完整实现后收录进 d2l 包，后续小节直接
``d2l.xxx`` 复用）的模式一致：每个组件在首次出现的 notebook 中给出完整
实现，并同步收录到本模块；后续小节直接 ``from src.attention import ...``
复用，避免多份重复定义（也避免修复 bug 时多处同步修改）。

各组件首次出现位置：
- ``masked_softmax`` / ``AdditiveAttention`` / ``DotProductAttention``：10.3 节
- ``transpose_qkv`` / ``transpose_output`` / ``MultiHeadAttention``：10.5 节
- ``PositionalEncoding``：10.6 节
- ``AttentionDecoder``：10.4 节

其中 ``masked_softmax`` 与 ``Decoder`` 复用 ``src.utils`` 中的既有实现
（``masked_softmax`` 的完整实现见 ``src/utils.py`` 及 10.3 节 notebook）。
"""

import math
import torch
from torch import nn

from src.pypto_ops import (PyPTOBMM, PyPTOTanh, PyPTOLinear,
                           PyPTOFusedAttention)
from src.utils import Decoder, masked_softmax


class AdditiveAttention(nn.Module):
    """加性注意力（PyPTO 算子版）"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = PyPTOLinear(key_size, num_hiddens, bias=False)
        self.W_q = PyPTOLinear(query_size, num_hiddens, bias=False)
        self.w_v = PyPTOLinear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，"键－值"对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = PyPTOTanh.apply(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，"键-值"对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，"键－值"对的个数，值的维度)
        return PyPTOBMM.apply(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """缩放点积注意力（PyPTO 算子版）"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self._fused_failed = False

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，"键－值"对的个数，d)
    # values的形状：(batch_size，"键－值"对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        seq = queries.shape[-2]
        cols = ((seq + 7) // 8) * 8
        # 无掩码且无 dropout 时走融合 kernel（QK^T+softmax+PV 单 kernel，
        # 实测比分解式快 13 倍；attention_weights 由 kernel 输出的 P 提供）。
        # 融合 kernel 的 vec tile 仅两档（cols≤160→128 行，否则 64 行）；
        # cols>320 时 softmax 的 ROWMAX 会超出 UB 预算（2×rows×cols×4B<192KB
        # 要求 cols<384，320 为保守界），编译失败会污染进程内后续 pypto
        # 调用（表现为卡死而非回退），故在 Python 层预判、直接走分解路径。
        if (valid_lens is None and not self._fused_failed and cols <= 320
                and (not self.training or self.dropout.p == 0)):
            try:
                out, p = PyPTOFusedAttention.apply(
                    queries, keys, values, 1.0 / math.sqrt(d))
                self.attention_weights = p
                return out
            except Exception:
                # kernel 编译失败（如形状/平台限制）则回退分解实现
                self._fused_failed = True
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = PyPTOBMM.apply(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return PyPTOBMM.apply(self.dropout(self.attention_weights), values)


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者"键－值"对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者"键－值"对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者"键－值"对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者"键－值"对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """多头注意力（PyPTO 算子版）"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = PyPTOLinear(query_size, num_hiddens, bias=bias)
        self.W_k = PyPTOLinear(key_size, num_hiddens, bias=bias)
        self.W_v = PyPTOLinear(value_size, num_hiddens, bias=bias)
        self.W_o = PyPTOLinear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者"键－值"对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者"键－值"对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 注册为 buffer：随模块 .to(device) 移动，避免前向中每次 CPU→NPU 拷贝
        # （该拷贝在 NPU 图捕获内不允许）
        self.register_buffer('P', torch.zeros((1, max_len, num_hiddens)))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


__all__ = [
    "masked_softmax",
    "AdditiveAttention",
    "DotProductAttention",
    "transpose_qkv",
    "transpose_output",
    "MultiHeadAttention",
    "PositionalEncoding",
    "AttentionDecoder",
]
