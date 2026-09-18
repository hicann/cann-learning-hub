import os
import sys
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import pandas as pd

def load_data_fashion_mnist(batch_size, resize=None):
    
    transform = [transforms.ToTensor()]
    if resize:
        transform.insert(0, transforms.Resize(resize))
    data_transform = transforms.Compose(transform)
    # Fashion MNIST dataset
    train_dataset = torchvision.datasets.FashionMNIST(root='../data/',
                                               train=True, 
                                               transform=data_transform,
                                               download=True)
    
    test_dataset = torchvision.datasets.FashionMNIST(root='../data/',
                                              train=False, 
                                              transform=data_transform,
                                              download=True)
    
    # Data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, 
                              shuffle=True)
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, 
                             shuffle=False)
    
    return train_loader, test_loader

def load_data_cifar10(batch_size, resize=None):
    
    transform = [transforms.ToTensor()]
    if resize:
        transform.insert(0, transforms.Resize(resize))
    data_transform = transforms.Compose(transform)
    # Fashion MNIST dataset
    train_dataset = torchvision.datasets.CIFAR10(root='../data/',
                                               train=True, 
                                               transform=data_transform,
                                               download=True);
    
    test_dataset = torchvision.datasets.CIFAR10(root='../data/',
                                              train=False, 
                                              transform=data_transform,
                                              download=True);
    
    # Data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, 
                              shuffle=True)
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, 
                             shuffle=False)
    
    return train_loader, test_loader

def write2csv(filename : str,
              data : list,
              name : str):
    """把 data 列表写入 csv 的 name 列. 若 csv 不存在则新建.

    若新旧列长度不一致则抛出 ValueError (避免静默 NaN 对齐).
    """
    parent = os.path.dirname(filename)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        df = pd.DataFrame()

    if len(df) > 0 and len(df) != len(data):
        # 允许不等长: 新列按较短长度写入, 不足部分 NaN 填充
        # (不同训练 epochs 不同时行数不同, 如 07.05 的 lr sweep 10 epochs vs 其他 20 epochs;
        #  draw_figures 的 plot/idxmax/idxmin 均容忍 NaN)
        col = pd.Series(data, index=range(len(data)))
        df[name] = col.reindex(df.index)
    else:
        df[name] = data
    df.to_csv(filename, index=False)
    # print('write '+ name + ' csv done!')



def draw_figures(pth:str,
                 label_list:list,
                 title:str = None):
    """从 csv 读取数据并绘制曲线. acc/loss 分支独立 (非互斥)."""
    d = pd.read_csv(pth)
    legend_list = []
    for x, y in label_list:
        plt.plot(d[x], d[y])
        legend_list.append(y)
    plt.xlabel('epoch')

    plt.legend(legend_list)
    plt.grid()
    bbox = dict(boxstyle="round", fc="0.8")
    arrowprops = dict(arrowstyle = "->", connectionstyle = "angle,angleA=0,angleB=90,rad=10")
    bias = 0
    if title:
        plt.title(title)
    if title and 'acc' in title:
        plt.ylabel('accuracy')
        bias = 0
        for x, y in label_list:
            max_index = d[y].idxmax()
            plt.annotate(('max acc of {} = {:.4f}'.format(y, d[y][max_index])),
                         (d[x][max_index], d[y][max_index]), xytext=(0, -50 - bias), textcoords='offset points',
                         bbox=bbox, arrowprops=arrowprops)
            bias += 30
    if title and 'loss' in title:
        plt.ylabel('loss')
        bias = 0
        for x, y in label_list:
            min_index = d[y].idxmin()
            plt.annotate( ('min loss of {} = {:.4f}'.format(y, d[y][min_index])),
                         (d[x][min_index], d[y][min_index]), xytext=(0, 50 + bias), textcoords='offset points',
                         bbox=bbox, arrowprops=arrowprops)
            bias += 30
    plt.show()


def train_models(net, train_loader, test_loader, epochs, lr,
                 net_type=None,
                 csv_prefix='ch7_01',
                 device=None,
                 optimizer_name='sgd',
                 momentum=0.0,
                 show_progress=True, progress_every=1):
    """通用训练循环 (从 07.01-07.05 等答案中提取的统一封装).

    参数:
        net: 待训练模型
        train_loader / test_loader: DataLoader
        epochs: 训练轮数
        lr: 学习率
        net_type: 模型类型标识, 写入 csv 时用作列名前缀 (None 则不写 csv)
        csv_prefix: csv 文件名前缀, 默认 'ch7_01', 跨章节时传入如 'ch7_02'
        device: 训练设备, 默认 None 时自动探测 'npu:0' or 'cpu'
        optimizer_name: 优化器选择, 'sgd' (默认) 或 'adam'.
            - 'sgd': 与 train_ch6_sgd 一致, 适合 PyPTO 深网 + xavier 初始化
            - 'adam': 与 train_ch6 一致, 适合浅层无 BN 网络
        momentum: SGD 动量 (默认 0)
        show_progress: 是否实时打印训练进度 (epoch/batch/loss/acc)
        progress_every: 每 N 个 epoch 打印一次汇总 (默认 1)
    """
    if device is None:
        device = torch.device('npu:0' if torch_npu_available() else 'cpu')
    if isinstance(device, str):
        device = torch.device(device)

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    loss = nn.CrossEntropyLoss()

    def init_weights(m):
        # 兼容 nn.Linear / nn.Conv2d 以及 PyPTOConv2d / PyPTOLinear
        # PyPTO 模块的 nn.Parameter 默认 kaiming_uniform, 在深网下首层
        # 梯度仅 ~1e-5 无法学习; 改为 xavier 后首层梯度恢复到 ~1e-2 量级.
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
        elif type(m).__name__ in ('PyPTOConv2d', 'PyPTOLinear'):
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device)

    epoch_list = []
    train_loss = []
    train_acc = []
    test_acc = []

    n_batches = max(1, len(train_loader))
    # 每 epoch 固定 5 个等分采样点 (与 batch_size 无关, 保证写入同一
    # csv 的各网络列行数一致; 最后一点恒为 n_batches-1)
    sample_idx = {int(n_batches * (i + 1) / 5) - 1 for i in range(5)} if n_batches > 5 else set(range(n_batches))
    # 进度显示间隔: 每 epoch 最多约 20 次刷新 (epoch 很长的网络也能看到过程变化)
    display_step = max(1, n_batches // 20)
    log_tag = f'[{net_type}]' if net_type else '[train]'
    last_test_acc = 0.0
    # 趋势状态: batch 级采样点与 epoch 级汇总值
    prev_loss = prev_acc = prev_test = None
    prev_eloss = prev_eacc = prev_etest = None

    t_start = time.time()
    for epoch in range(epochs):
        loss_sum = 0.0
        acc = 0
        n_seen = 0
        net.train()
        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            bs = x.size(0)
            optimizer.zero_grad()
            predict = net(x)
            l = loss(predict, y)
            l.backward()
            optimizer.step()
            loss_sum += l.item() * bs
            acc += predict.max(dim=1)[1].eq(y).sum().item()
            n_seen += bs
            # 实时进度 (按 batch 间隔刷新, 与 csv 采样点独立)
            if show_progress and ((idx + 1) % display_step == 0 or idx == n_batches - 1):
                _loss = loss_sum / n_seen
                _acc = acc / n_seen
                elapsed = time.time() - t_start
                eta = elapsed / (epoch * n_batches + idx + 1) * (epochs * n_batches - (epoch * n_batches + idx + 1))
                done = (epoch * n_batches + idx + 1) / (epochs * n_batches)
                t_l = _trend(prev_loss, _loss, good_high=False)
                t_a = _trend(prev_acc, _acc, good_high=True)
                t_t = _trend(prev_test, last_test_acc, good_high=True)
                prev_loss, prev_acc = _loss, _acc
                _render_progress(
                    f'{log_tag} |{_progress_bar(done)}| {done * 100:5.1f}% '
                    f'epoch {epoch + 1}/{epochs} batch {idx + 1}/{n_batches} '
                    f'loss={_loss:.4f} {_mini_bar(_loss, good_high=False)} {t_l} '
                    f'acc={_acc:.4f} {_mini_bar(_acc)} {t_a} '
                    f'test={last_test_acc:.4f} {_mini_bar(last_test_acc)} {t_t} '
                    f'et={_fmt_time(elapsed)} eta={_fmt_time(eta)}'
                )
            # 写入 csv 用的 train_loss/train_acc 收集 (与进度采样点一致)
            if idx in sample_idx:
                loss_now = loss_sum / n_seen
                acc_now = acc / n_seen
                train_acc.append(acc_now)
                train_loss.append(loss_now)
                epoch_list.append(epoch + (idx + 1) / n_batches)

        # 评估 (不计算 CE loss, 只统计 accuracy)
        acc = 0
        net.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                predict = net(x)
                acc += predict.max(dim=1)[1].eq(y).sum().item()
        acc /= len(test_loader.dataset)
        test_acc.append(acc)
        last_test_acc = acc
        prev_test = acc
        if show_progress and ((epoch + 1) % progress_every == 0 or epoch == epochs - 1):
            done = (epoch + 1) / epochs
            t_l = _trend(prev_eloss, train_loss[-1], good_high=False)
            t_a = _trend(prev_eacc, train_acc[-1], good_high=True)
            t_t = _trend(prev_etest, test_acc[-1], good_high=True)
            prev_eloss, prev_eacc, prev_etest = train_loss[-1], train_acc[-1], test_acc[-1]
            _render_progress(
                f'{log_tag} |{_progress_bar(done)}| {done * 100:5.1f}% '
                f'epoch {epoch + 1}/{epochs} '
                f'train_loss={train_loss[-1]:.4f} {_mini_bar(train_loss[-1], good_high=False)} {t_l} '
                f'train_acc={train_acc[-1]:.4f} {_mini_bar(train_acc[-1])} {t_a} '
                f'test_acc={test_acc[-1]:.4f} {_mini_bar(test_acc[-1])} {t_t} '
                f'et={_fmt_time(time.time() - t_start)}'
            )

    if show_progress:
        _render_progress(
            f'{log_tag} 训练完成: '
            f'train_loss={train_loss[-1]:.4f} {_mini_bar(train_loss[-1], good_high=False)} '
            f'train_acc={train_acc[-1]:.4f} {_mini_bar(train_acc[-1])} '
            f'test_acc={test_acc[-1]:.4f} {_mini_bar(test_acc[-1])} '
            f'et={_fmt_time(time.time() - t_start)}'
        )

    if net_type is not None:
        # 跨章节支持: csv_prefix 控制写入哪个 csv 文件
        pth_csv1 = os.path.join('../data/ch07_output/', f'{csv_prefix}.csv')
        pth_csv2 = os.path.join('../data/ch07_output/', f'{csv_prefix}_eval.csv')
        # epoch_test 与 test_acc 长度对齐 (都按 epoch 数)
        epoch_test = list(range(1, len(test_acc) + 1))
        write2csv(pth_csv1, epoch_list, net_type + '_epoch_train')
        write2csv(pth_csv1, train_loss, net_type + '_train_loss')
        write2csv(pth_csv1, train_acc, net_type + '_train_acc')
        write2csv(pth_csv2, epoch_test, net_type + '_epoch_test')
        write2csv(pth_csv2, test_acc, net_type + '_test_acc')
    else:
        print('net_type is None, don\'t record the data')


def _in_notebook() -> bool:
    """检测是否运行在 Jupyter notebook 环境."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and 'ZMQInteractiveShell' in type(ip).__name__
    except Exception:
        return False


_IN_NOTEBOOK = _in_notebook()


def _progress_bar(frac: float, width: int = 20) -> str:
    filled = int(round(width * min(1.0, max(0.0, frac))))
    return '█' * filled + '░' * (width - filled)


def _mini_bar(value, good_high: bool = True, width: int = 8) -> str:
    """迷你横条直观显示数值大小. good_high=True 时值越大条越长 (acc 等),
    False 时值越小条越长 (loss 等)."""
    v = min(1.0, max(0.0, value))
    if not good_high:
        v = 1.0 - v
    filled = int(round(width * v))
    return '█' * filled + '░' * (width - filled)


def _trend(prev, cur, good_high: bool = True) -> str:
    """趋势箭头: ↑ 变好, ↓ 变差, → 持平, · 无历史数据.
    good_high=True 表示值越大越好 (acc), False 表示值越小越好 (loss)."""
    if prev is None:
        return '·'
    diff = cur - prev
    if abs(diff) < 1e-4:
        return '→'
    better = diff > 0 if good_high else diff < 0
    return '↑' if better else '↓'


def _render_progress(line: str) -> None:
    """以单行覆盖方式输出进度, Jupyter 与终端通用."""
    if _IN_NOTEBOOK:
        from IPython.display import clear_output
        clear_output(wait=True)
        print(line)
    else:
        sys.stdout.write('\r' + line.ljust(100))
        sys.stdout.flush()


def _fmt_time(seconds):
    """把秒数格式化为 mm:ss (或 hh:mm:ss). 负数返回 '00:00'."""
    seconds = max(0, seconds)
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f'{h:d}:{m:02d}:{s:02d}'
    return f'{m:02d}:{s:02d}'


def torch_npu_available():
    try:
        import torch_npu
        return torch_npu.npu.is_available()
    except (ImportError, RuntimeError, OSError):
        return False