"""纯 PyTorch 公共工具模块（替代 d2l 包中第8/9章用到的函数）。

提供 d2l 常用函数的纯 PyTorch 等价实现，供第8/9章 notebook 通过
``from src.utils import ...`` 复用，避免依赖 d2l 包。

覆盖函数：
- 文本处理：read_time_machine, tokenize, Vocab
- 数据迭代：seq_data_iter_random, seq_data_iter_sequential, SeqDataLoader
- 数据加载：load_corpus_time_machine, load_data_time_machine
- 机器翻译：read_data_nmt, preprocess_nmt, tokenize_nmt, truncate_pad,
            build_array_nmt, load_data_nmt
- 可视化：Animator, plot, show_list_len_pair_hist
- 训练辅助：Timer, Accumulator, sgd, synthetic_data
- 设备选择：try_gpu
- 序列工具：sequence_mask, bleu
- 基类：Encoder, Decoder, EncoderDecoder
- RNN 模型封装：RNNModel, RNNModelScratch（替代 d2l 的 RNNModel/RNNModelScratch）
"""

import collections
import hashlib
import math
import numpy as np
import os
import random
import re
import shutil
import tarfile
import time
import urllib.request
import zipfile

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from IPython import display

# =========================================================================
# 设备选择
# =========================================================================


def try_gpu(i=0):
    """返回 gpu(i) 如果存在，否则返回 cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")

# =========================================================================
# 数据下载
# =========================================================================


DATA_HUB = {}
DATA_URL = "https://d2l-data.s3-accelerate.amazonaws.com/"


def download(name, cache_dir=os.path.join("..", "data")):
    """下载 DATA_HUB 中的文件并返回本地文件名。"""
    assert name in DATA_HUB, f"{name} 不在 DATA_HUB 中"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, "rb") as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f"正在从 {url} 下载 {fname}...")
    r = urllib.request.urlopen(url)
    with open(fname, "wb") as f:
        f.write(r.read())
    return fname


def download_extract(name, folder=None):
    """下载并提取 zip/tar 文件。"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == ".zip":
        fp = zipfile.ZipFile(fname, "r")
    elif ext in (".tar", ".gz"):
        fp = tarfile.open(fname, "r")
    else:
        assert False, "只有 zip/tar 文件可以被解压缩"
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


DATA_HUB["time_machine"] = (
    DATA_URL + "timemachine.txt",
    "090b5e7e70c295757f55df93cb0a180b9691891a",
)
DATA_HUB["fra-eng"] = (
    DATA_URL + "fra-eng.zip",
    "94646ad1522d915e7b0f9296181140edcf86a4f5",
)

# =========================================================================
# 文本预处理
# =========================================================================


def read_time_machine():
    """将时间机器数据集加载到文本行的列表中。"""
    with open(download("time_machine"), "r") as f:
        lines = f.readlines()
    return [re.sub("[^A-Za-z]+", " ", line).strip().lower() for line in lines]


def tokenize(lines, token="word"):
    """将文本行拆分为词元列表。

    Args:
        lines: 字符串列表
        token: "word" 表示按空格分词，"char" 表示按字符分词

    Returns:
        词元列表的列表
    """
    if token == "word":
        return [line.split() for line in lines]
    elif token == "char":
        return [list(line) for line in lines]
    else:
        raise ValueError(f"未知的词元类型：{token}")


class Vocab:
    """文本词表。"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        self.idx_to_token = ["<unk>"] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    """统计词元的频率。"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# =========================================================================
# 序列数据迭代器
# =========================================================================


def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列。

    Args:
        corpus: 整数索引列表
        batch_size: 批量大小
        num_steps: 时间步数

    Yields:
        (X, Y): 特征和标签张量
    """
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列。

    Args:
        corpus: 整数索引列表
        batch_size: 批量大小
        num_steps: 时间步数

    Yields:
        (X, Y): 特征和标签张量
    """
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:
    """加载序列数据的迭代器。"""

    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表。"""
    lines = read_time_machine()
    tokens = tokenize(lines, "char")
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表。"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

# =========================================================================
# 机器翻译数据处理
# =========================================================================


def read_data_nmt():
    """载入"英语－法语"数据集。"""
    data_dir = download_extract("fra-eng")
    with open(os.path.join(data_dir, "fra.txt"), "r",
              encoding="utf-8") as f:
        return f.read()


def preprocess_nmt(text):
    """预处理"英语－法语"数据集。

    用空格替换不间断空格，小写，并在标点前加空格。
    """
    def no_space(char, prev_char):
        return char in set(",.!?") and prev_char != " "

    text = text.replace("\u202f", " ").replace("\xa0", " ").lower()
    out = [" " + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return "".join(out)


def tokenize_nmt(text, num_examples=None):
    """词元化"英语－法语"数据集。

    Returns:
        source: 源语言词元列表的列表
        target: 目标语言词元列表的列表
    """
    source, target = [], []
    for i, line in enumerate(text.split("\n")):
        if num_examples and i > num_examples:
            break
        parts = line.split("\t")
        if len(parts) == 2:
            source.append(parts[0].split(" "))
            target.append(parts[1].split(" "))
    return source, target


def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列。"""
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量。"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab["<eos>"]] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab["<pad>"]) for l in lines])
    valid_len = (array != vocab["<pad>"]).type(torch.int32).sum(1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表。"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2,
                      reserved_tokens=["<pad>", "<bos>", "<eos>"])
    tgt_vocab = Vocab(target, min_freq=2,
                      reserved_tokens=["<pad>", "<bos>", "<eos>"])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图。"""
    _, _, patches = plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch("/")
    plt.legend(legend)

# =========================================================================
# 计时器
# =========================================================================


class Timer:
    """记录多次运行时间。"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return torch.tensor(self.times).cumsum(dim=0).tolist()

# =========================================================================
# 累加器
# =========================================================================


class Accumulator:
    """在 n 个变量上累加求和。"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# =========================================================================
# 可视化
# =========================================================================


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置 matplotlib 坐标轴的属性。"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if xlim:
        axes.set_xlim(xlim)
    if ylim:
        axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def use_svg_display():
    """使用 PNG 格式在 notebook 中显示图形。"""
    try:
        import matplotlib_inline
        matplotlib_inline.backend_inline.set_matplotlib_formats("png")
    except (ImportError, AttributeError):
        pass


class Animator:
    """在动画中绘制数据。"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale="linear", yscale="linear",
                 fmts=("-", "m--", "g-.", "r:"), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """向图表中添加多个数据点。"""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x_vals, y_vals, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_vals, y_vals, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale="linear", yscale="linear",
         fmts=("-", "m--", "g-.", "r:"), figsize=(3.5, 2.5), axes=None):
    """绘制数据点。"""
    if legend is None:
        legend = []

    def _set_axes(ax):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if legend:
            ax.legend(legend)
        ax.grid()

    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=figsize)
    axes.cla()
    if Y is not None:
        if isinstance(Y, (list, tuple)):
            Y = [y.detach().cpu().numpy() if torch.is_tensor(y) else y
                 for y in Y]
        elif torch.is_tensor(Y):
            Y = Y.detach().cpu().numpy()
        if isinstance(X, (list, tuple)):
            X = [x.detach().cpu().numpy() if torch.is_tensor(x) else x
                 for x in X]
        elif torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        if isinstance(X, (list, tuple)) and isinstance(Y, (list, tuple)):
            for x_i, y_i in zip(X, Y):
                axes.plot(x_i, y_i)
        else:
            axes.plot(X, *Y) if isinstance(Y, list) else axes.plot(X, Y)
    else:
        if isinstance(X, (list, tuple)):
            if len(X) > 0 and isinstance(X[0], (list, tuple, np.ndarray)):
                for x_i in X:
                    if torch.is_tensor(x_i):
                        x_i = x_i.detach().cpu().numpy()
                    axes.plot(x_i)
            elif torch.is_tensor(X[0]) if len(X) > 0 else False:
                X = [x_i.detach().cpu().numpy() if torch.is_tensor(x_i) else x_i
                     for x_i in X]
                for x_i in X:
                    axes.plot(x_i)
            else:
                axes.plot(list(X))
        else:
            if torch.is_tensor(X):
                X = X.detach().cpu().numpy()
            axes.plot(X)
    _set_axes(axes)
    plt.show()

# =========================================================================
# 优化
# =========================================================================


def sgd(params, lr, batch_size):
    """小批量随机梯度下降。"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def synthetic_data(w, b, num_examples, device=None):
    """生成 y = Xw + b + noise 的合成数据。"""
    X = torch.normal(0, 1, (num_examples, len(w)), device=device)
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape, device=device)
    return X, y.reshape((-1, 1))

# =========================================================================
# 数据加载辅助
# =========================================================================


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器。"""
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)


# =========================================================================
# 模型评估
# =========================================================================


def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失。"""
    metric = Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def _accuracy(y_hat, y):
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# =========================================================================
# RNN 预测
# =========================================================================


def predict_ch8(prefix, num_preds, net, vocab, device):
    """在 prefix 后生成新字符。

    Args:
        prefix:   字符串前缀
        num_preds: 要生成的字符数
        net:      RNN 模型（有 begin_state 方法）
        vocab:    词表
        device:   设备

    Returns:
        生成的字符串
    """
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape(
        (1, 1))

    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])

    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))

    return "".join([vocab.idx_to_token[i] for i in outputs])


# =========================================================================
# 梯度裁剪
# =========================================================================


def grad_clipping(net, theta):
    """裁剪梯度。

    Args:
        net:   模型（nn.Module 或有 .params 属性）
        theta: 裁剪阈值
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            scale = (theta / norm).to(device=param.grad.device, dtype=param.grad.dtype)
            param.grad[:] = param.grad[:] * scale


# =========================================================================
# RNN 训练
# =========================================================================


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（第8章风格）。

    Args:
        net:             模型
        train_iter:      训练数据迭代器
        loss:            损失函数
        updater:         优化器 或 自定义更新函数
        device:          设备
        use_random_iter: 是否使用随机采样

    Returns:
        (困惑度, 速度)
    """
    state, timer = None, Timer()
    metric = Accumulator(2)

    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state = state.detach()
            else:
                state = tuple(s.detach() for s in state)

        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long())

        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)

        metric.add(l * y.numel(), y.numel())

    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False, use_plot=True, loss_fn=None,
              verbose=False):
    """训练 RNN 模型（第8章风格）。

    Args:
        net:             模型
        train_iter:      训练数据迭代器
        vocab:           词表
        lr:              学习率
        num_epochs:      迭代周期数
        device:          设备
        use_random_iter: 是否使用随机采样
        use_plot:        是否使用 Animator 绘图（CLI 环境建议 False）
        loss_fn:         可选，自定义损失函数（签名 loss(logits, labels) → 标量）
                          默认 nn.CrossEntropyLoss()
        verbose:         是否打印中间训练信息（每 10 epoch 的预测文本和困惑度）。
                          默认 False，仅输出最终结果（困惑度、速度和最终预测文本），
                          避免训练过程产生冗余输出。
    """
    loss = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
    animator = None
    if use_plot:
        animator = Animator(xlabel="epoch", ylabel="perplexity",
                            legend=["train"], xlim=[10, num_epochs])

    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)

    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)

    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if verbose and (epoch + 1) % 10 == 0:
            print(predict("time traveller"))
            if animator is not None:
                animator.add(epoch + 1, [ppl])
            else:
                print(f"  [epoch {epoch+1}/{num_epochs}] perplexity={ppl:.1f}")

    print(f"困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}")
    print(predict("time traveller"))
    print(predict("traveller"))


# =========================================================================
# RNN 模型封装（替代 d2l 的 RNNModel / RNNModelScratch）
# =========================================================================


class RNNModelScratch:
    """从零实现的 RNN 模型（第 8.5 节风格）。

    与 d2l.RNNModelScratch 兼容：接收 get_params / init_state / forward_fn，
    用 one-hot 编码输入，forward_fn 返回 (输出, 新状态)。
    """

    def __init__(self, vocab_size, num_hiddens, device, get_params,
                 init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


class RNNModel(nn.Module):
    """用 nn.RNN / nn.GRU / nn.LSTM 隐层 + 输出层封装的语言模型（第 8.6 节风格）。

    与 d2l.RNNModel 兼容：接收一个 rnn_layer（有 hidden_size / num_layers /
    bidirectional 属性），输出层为 nn.Linear。
    """

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size).type(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device)
        else:
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device))


# =========================================================================
# 序列掩码工具
# =========================================================================


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项。

    Args:
        X: 输入张量
        valid_len: 有效长度 (batch_size,) 类型张量
        value: 填充值

    Returns:
        原地修改后的 X
    """
    maxlen = X.size(1)
    mask = torch.arange(maxlen, dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def bleu(pred_seq, label_seq, k):
    """计算 BLEU 分数。

    Args:
        pred_seq: 预测序列字符串（空格分隔）
        label_seq: 标签序列字符串（空格分隔）
        k: 最长 n-gram 匹配数

    Returns:
        BLEU 分数
    """
    pred_tokens, label_tokens = pred_seq.split(" "), label_seq.split(" ")
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[" ".join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[" ".join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[" ".join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


# =========================================================================
# 编码器-解码器基类
# =========================================================================


class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类。"""

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
