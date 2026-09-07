"""Polar 编码器与 SC（逐次消去）解码器 -- TXS-10002-2025 标准 6.9.1.4 节。"""


__all__ = [
    "RATE_TABLE",
    "RELIABILITY_SEQ_1024",
    "VALID_CODE_LENGTHS",
    "PolarDecoder",
    "PolarEncoder",
    "get_info_bit_count",
    "get_polar_decoder",
]


from functools import lru_cache

import numpy as np

try:
    from nearlink_sdr_accel import RustPolarDecoder as _RustPolarDecoder
    from nearlink_sdr_accel import RustPolarEncoder as _RustPolarEncoder
    _HAS_RUST_ACCEL = True
except ImportError:
    _HAS_RUST_ACCEL = False

# N_max=1024 的可靠性序列 Q_0^{N_max-1}
# 按可靠性升序排列，索引 i 对应比特索引 Q_i。
RELIABILITY_SEQ_1024 = [
    0,    1,    2,    4,    8,    16,   32,   3,    5,    64,   9,    6,    17,   10,   18,   128,
    12,   33,   65,   20,   256,  34,   24,   36,   7,    129,  66,   512,  11,   40,   68,   130,
    19,   13,   48,   14,   72,   257,  21,   132,  35,   258,  26,   513,  80,   37,   25,   22,
    136,  260,  264,  38,   514,  96,   67,   41,   144,  28,   69,   42,   516,  49,   74,   272,
    160,  520,  288,  528,  192,  544,  70,   44,   131,  81,   50,   73,   15,   320,  133,  52,
    23,   134,  384,  76,   137,  82,   56,   27,   97,   39,   259,  84,   138,  145,  261,  29,
    43,   98,   515,  88,   140,  30,   146,  71,   262,  265,  161,  576,  45,   100,  640,  51,
    148,  46,   75,   266,  273,  517,  104,  162,  53,   193,  152,  77,   164,  768,  268,  274,
    518,  54,   83,   57,   521,  112,  135,  78,   289,  194,  85,   276,  522,  58,   168,  139,
    99,   86,   60,   280,  89,   290,  529,  524,  196,  141,  101,  147,  176,  142,  530,  321,
    31,   200,  90,   545,  292,  322,  532,  263,  149,  102,  105,  304,  296,  163,  92,   47,
    267,  385,  546,  324,  208,  386,  150,  153,  165,  106,  55,   328,  536,  577,  548,  113,
    154,  79,   269,  108,  578,  224,  166,  519,  552,  195,  270,  641,  523,  275,  580,  291,
    59,   169,  560,  114,  277,  156,  87,   197,  116,  170,  61,   531,  525,  642,  281,  278,
    526,  177,  293,  388,  91,   584,  769,  198,  172,  120,  201,  336,  62,   282,  143,  103,
    178,  294,  93,   644,  202,  592,  323,  392,  297,  770,  107,  180,  151,  209,  284,  648,
    94,   204,  298,  400,  608,  352,  325,  533,  155,  210,  305,  547,  300,  109,  184,  534,
    537,  115,  167,  225,  326,  306,  772,  157,  656,  329,  110,  117,  212,  171,  776,  330,
    226,  549,  538,  387,  308,  216,  416,  271,  279,  158,  337,  550,  672,  118,  332,  579,
    540,  389,  173,  121,  553,  199,  784,  179,  228,  338,  312,  704,  390,  174,  554,  581,
    393,  283,  122,  448,  353,  561,  203,  63,   340,  394,  527,  582,  556,  181,  295,  285,
    232,  124,  205,  182,  643,  562,  286,  585,  299,  354,  211,  401,  185,  396,  344,  586,
    645,  593,  535,  240,  206,  95,   327,  564,  800,  402,  356,  307,  301,  417,  213,  568,
    832,  588,  186,  646,  404,  227,  896,  594,  418,  302,  649,  771,  360,  539,  111,  331,
    214,  309,  188,  449,  217,  408,  609,  596,  551,  650,  229,  159,  420,  310,  541,  773,
    610,  657,  333,  119,  600,  339,  218,  368,  652,  230,  391,  313,  450,  542,  334,  233,
    555,  774,  175,  123,  658,  612,  341,  777,  220,  314,  424,  395,  673,  583,  355,  287,
    183,  234,  125,  557,  660,  616,  342,  316,  241,  778,  563,  345,  452,  397,  403,  207,
    674,  558,  785,  432,  357,  187,  236,  664,  624,  587,  780,  705,  126,  242,  565,  398,
    346,  456,  358,  405,  303,  569,  244,  595,  189,  566,  676,  361,  706,  589,  215,  786,
    647,  348,  419,  406,  464,  680,  801,  362,  590,  409,  570,  788,  597,  572,  219,  311,
    708,  598,  601,  651,  421,  792,  802,  611,  602,  410,  231,  688,  653,  248,  369,  190,
    364,  654,  659,  335,  480,  315,  221,  370,  613,  422,  425,  451,  614,  543,  235,  412,
    343,  372,  775,  317,  222,  426,  453,  237,  559,  833,  804,  712,  834,  661,  808,  779,
    617,  604,  433,  720,  816,  836,  347,  897,  243,  662,  454,  318,  675,  618,  898,  781,
    376,  428,  665,  736,  567,  840,  625,  238,  359,  457,  399,  787,  591,  678,  434,  677,
    349,  245,  458,  666,  620,  363,  127,  191,  782,  407,  436,  626,  571,  465,  681,  246,
    707,  350,  599,  668,  790,  460,  249,  682,  573,  411,  803,  789,  709,  365,  440,  628,
    689,  374,  423,  466,  793,  250,  371,  481,  574,  413,  603,  366,  468,  655,  900,  805,
    615,  684,  710,  429,  794,  252,  373,  605,  848,  690,  713,  632,  482,  806,  427,  904,
    414,  223,  663,  692,  835,  619,  472,  455,  796,  809,  714,  721,  837,  716,  864,  810,
    606,  912,  722,  696,  377,  435,  817,  319,  621,  812,  484,  430,  838,  667,  488,  239,
    378,  459,  622,  627,  437,  380,  818,  461,  496,  669,  679,  724,  841,  629,  351,  467,
    438,  737,  251,  462,  442,  441,  469,  247,  683,  842,  738,  899,  670,  783,  849,  820,
    728,  928,  791,  367,  901,  630,  685,  844,  633,  711,  253,  691,  824,  902,  686,  740,
    850,  375,  444,  470,  483,  415,  485,  905,  795,  473,  634,  744,  852,  960,  865,  693,
    797,  906,  715,  807,  474,  636,  694,  254,  717,  575,  913,  798,  811,  379,  697,  431,
    607,  489,  866,  723,  486,  908,  718,  813,  476,  856,  839,  725,  698,  914,  752,  868,
    819,  814,  439,  929,  490,  623,  671,  739,  916,  463,  843,  381,  497,  930,  821,  726,
    961,  872,  492,  631,  729,  700,  443,  741,  845,  920,  382,  822,  851,  730,  498,  880,
    742,  445,  471,  635,  932,  687,  903,  825,  500,  846,  745,  826,  732,  446,  962,  936,
    475,  853,  867,  637,  907,  487,  695,  746,  828,  753,  854,  857,  504,  799,  255,  964,
    909,  719,  477,  915,  638,  748,  944,  869,  491,  699,  754,  858,  478,  968,  383,  910,
    815,  976,  870,  917,  727,  493,  873,  701,  931,  756,  860,  499,  731,  823,  922,  874,
    918,  502,  933,  743,  760,  881,  494,  702,  921,  501,  876,  847,  992,  447,  733,  827,
    934,  882,  937,  963,  747,  505,  855,  924,  734,  829,  965,  938,  884,  506,  749,  945,
    966,  755,  859,  940,  830,  911,  871,  639,  888,  479,  946,  750,  969,  508,  861,  757,
    970,  919,  875,  862,  758,  948,  977,  923,  972,  761,  877,  952,  495,  703,  935,  978,
    883,  762,  503,  925,  878,  735,  993,  885,  939,  994,  980,  926,  764,  941,  967,  886,
    831,  947,  507,  889,  984,  751,  942,  996,  971,  890,  509,  949,  973,  1000, 892,  950,
    863,  759,  1008, 510,  979,  953,  763,  974,  954,  879,  981,  982,  927,  995,  765,  956,
    887,  985,  997,  986,  943,  891,  998,  766,  511,  988,  1001, 951,  1002, 893,  975,  894,
    1009, 955,  1004, 1010, 957,  983,  958,  987,  1012, 999,  1016, 767,  989,  1003, 990,  1005,
    959,  1011, 1013, 895,  1006, 1014, 1017, 1018, 991,  1020, 1007, 1015, 1019, 1021, 1022, 1023]

# Rate matching table (table 23): rate_str -> {N: K}
RATE_TABLE = {
    "1/4": {1024: 256, 512: 96, 256: 48, 128: 24, 64: 12},
    "3/8": {1024: 384, 512: 160, 256: 80, 128: 40, 64: 20},
    "1/2": {1024: 512, 512: 224, 256: 112, 128: 56, 64: 28},
    "5/8": {1024: 640, 512: 288, 256: 144, 128: 72, 64: 36},
    "3/4": {1024: 768, 512: 352, 256: 176, 128: 88, 64: 44},
    "7/8": {1024: 896, 512: 416, 256: 208, 128: 104, 64: 52},
}

VALID_CODE_LENGTHS = {64, 128, 256, 512, 1024}


def get_info_bit_count(rate_str: str, N: int) -> int:
    """从速率匹配表中查询给定码率和码长对应的信息位数 K。"""
    if rate_str not in RATE_TABLE:
        raise ValueError(f"Unsupported rate '{rate_str}'. Valid: {list(RATE_TABLE.keys())}")
    if N not in RATE_TABLE[rate_str]:
        raise ValueError(f"Unsupported code length {N} for rate '{rate_str}'.")
    return RATE_TABLE[rate_str][N]


@lru_cache(maxsize=16)
def _get_reliability_sequence(N: int) -> tuple[int, ...]:
    """从 N_max=1024 可靠性序列中提取码长 N 的可靠性序列。

    过滤比特索引 < N 的条目，保持可靠性顺序不变。
    """
    return tuple(q for q in RELIABILITY_SEQ_1024 if q < N)


def _get_frozen_and_info_sets(N: int, K: int) -> tuple[set[int], set[int]]:
    """确定冻结位和信息位的位置集合。

    可靠性序列中最后 K 个（最可靠）为信息位，
    前 N-K 个（最不可靠）为冻结位。
    """
    seq = _get_reliability_sequence(N)
    frozen_set = set(seq[: N - K])
    info_set = set(seq[N - K :])
    return frozen_set, info_set


class PolarEncoder:
    """使用 GF(2) 上蝶形（递归）算法的 Polar 编码器。"""

    def __init__(self, N: int, K: int) -> None:
        if N not in VALID_CODE_LENGTHS:
            raise ValueError(f"N must be in {sorted(VALID_CODE_LENGTHS)}, got {N}")
        if K >= N or K < 1:
            raise ValueError(f"K must satisfy 1 <= K < N, got K={K}, N={N}")
        self.N = N
        self.K = K
        self.frozen_set, self.info_set = _get_frozen_and_info_sets(N, K)
        seq = _get_reliability_sequence(N)
        self.info_positions = sorted(seq[N - K :])
        self._info_pos_arr = np.array(self.info_positions, dtype=np.intp)
        # Rust 加速
        self._rust_encoder = None
        if _HAS_RUST_ACCEL:
            self._rust_encoder = _RustPolarEncoder(
                N, np.array(self.info_positions, dtype=np.int64),
            )

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """将 K 个信息位编码为 N 个编码位。

        :param info_bits: 长度为 K 的比特数组（取值 0 或 1）。

        :returns: 长度为 N 的码字（取值 0 或 1）。
        """
        info_bits = np.asarray(info_bits, dtype=np.int8)
        if info_bits.shape != (self.K,):
            raise ValueError(f"Expected {self.K} info bits, got shape {info_bits.shape}")

        if self._rust_encoder is not None:
            return np.asarray(self._rust_encoder.encode(info_bits), dtype=np.int8)

        # Build encoder input u — vectorized insert
        u = np.zeros(self.N, dtype=np.int8)
        u[self._info_pos_arr] = info_bits

        # Butterfly encoding: vectorized GF(2) transform
        d = u.copy()
        stage = 1
        while stage < self.N:
            for j in range(0, self.N, 2 * stage):
                d[j:j + stage] ^= d[j + stage:j + 2 * stage]
            stage <<= 1

        return d


class PolarDecoder:
    """Successive Cancellation (SC) decoder in LLR domain。

    使用预分配的二维数组存储中间 LLR 和 partial sums, 避免递归中频繁分配内存。
    """

    def __init__(self, N: int, K: int) -> None:
        if N not in VALID_CODE_LENGTHS:
            raise ValueError(f"N must be in {sorted(VALID_CODE_LENGTHS)}, got {N}")
        if K >= N or K < 1:
            raise ValueError(f"K must satisfy 1 <= K < N, got K={K}, N={N}")
        self.N = N
        self.K = K
        self.n = int(np.log2(N))
        self.frozen_set, self.info_set = _get_frozen_and_info_sets(N, K)
        seq = _get_reliability_sequence(N)
        self.info_positions = sorted(seq[N - K :])
        self._info_pos_arr = np.array(self.info_positions, dtype=np.intp)
        self._is_frozen = np.ones(N, dtype=bool)
        for pos in self.info_positions:
            self._is_frozen[pos] = False
        # 预分配 (n+1) 层, 每层 N 元素
        self._L = np.zeros((self.n + 1, N), dtype=np.float64)
        self._B = np.zeros((self.n + 1, N), dtype=np.int8)
        # SSC: 预计算每个子树的类型 (rate-0 / rate-1 / partial)
        self._node_type = self._build_node_types()
        # Rust 加速: 可用时创建 Rust 解码器实例
        self._rust_decoder = None
        if _HAS_RUST_ACCEL:
            self._rust_decoder = _RustPolarDecoder(
                self.n,
                self._is_frozen,
                np.array(self.info_positions, dtype=np.int64),
            )

    def _build_node_types(self) -> dict[tuple[int, int], int]:
        """预计算所有子树的类型: 0=rate-0, 1=rate-1, 2=partial。

        使用前缀和做 O(1) 范围查询, 迭代遍历避免递归开销。
        rate-0/rate-1 子树不再向下遍历, 与 SSC 剪枝对齐。
        """
        N = self.N
        frozen = self._is_frozen
        # 前缀和: prefix[i] = sum(frozen[0:i])
        prefix = np.empty(N + 1, dtype=np.int32)
        prefix[0] = 0
        np.cumsum(frozen, out=prefix[1:])

        node_type: dict[tuple[int, int], int] = {}
        stack = [(0, N)]
        while stack:
            start, length = stack.pop()
            cnt = int(prefix[start + length] - prefix[start])
            if cnt == length:
                node_type[(start, length)] = 0
            elif cnt == 0:
                node_type[(start, length)] = 1
            else:
                node_type[(start, length)] = 2
                if length > 1:
                    half = length >> 1
                    stack.append((start, half))
                    stack.append((start + half, half))
        return node_type

    def decode(self, llr: np.ndarray) -> np.ndarray:
        """使用 SC 算法将 N 个信道 LLR 解码为 K 个信息位。

        约定：正 LLR 表示比特 0 的可能性更高。
        """
        llr = np.asarray(llr, dtype=np.float64)
        if llr.shape != (self.N,):
            raise ValueError(f"Expected {self.N} LLRs, got shape {llr.shape}")

        if self._rust_decoder is not None:
            return np.asarray(self._rust_decoder.decode(llr), dtype=np.int8)

        self._L[0, :self.N] = llr
        self._sc(0, self.N, 0)
        return self._B[self.n, :self.N][self._info_pos_arr].copy()

    def _sc(self, start: int, length: int, depth: int) -> None:
        """SSC 优化的 SC 解码。

        rate-0 子树: 所有位为冻结位, 直接置零。
        rate-1 子树: 硬判决 LLR 得到编码位, 极性变换得到信息位。
        partial 子树: 标准 f/g 递归。
        """
        L = self._L
        B = self._B

        node_t = self._node_type.get((start, length), 2)

        if node_t == 0:
            # rate-0: 全冻结 → 解码位和部分和均为零
            end = start + length
            B[self.n, start:end] = 0
            B[depth, start:end] = 0
            return

        if node_t == 1:
            # rate-1: 全信息位
            end = start + length
            # 硬判决得到该层的编码位 x
            x = np.where(L[depth, start:end] < 0, 1, 0).astype(np.int8)
            # 部分和 = 编码位 (供父节点 g 操作和合并使用)
            B[depth, start:end] = x
            if length == 1:
                B[self.n, start] = x[0]
            else:
                # 极性变换 x → u (F^⊗n 自逆)
                u = x.copy()
                step = length
                while step >= 2:
                    half_s = step >> 1
                    u_view = u.reshape(-1, step)
                    u_view[:, :half_s] ^= u_view[:, half_s:]
                    step >>= 1
                B[self.n, start:end] = u
            return

        # partial: 标准 SC 递归
        if length == 1:
            if self._is_frozen[start]:
                B[self.n, start] = 0
            else:
                B[self.n, start] = 0 if L[depth, start] >= 0.0 else 1
            return

        half = length >> 1
        mid = start + half
        end = start + length
        d1 = depth + 1

        # f 操作
        a = L[depth, start:mid]
        b = L[depth, mid:end]
        dst = L[d1, start:mid]
        np.minimum(np.abs(a), np.abs(b), out=dst)
        dst *= np.sign(a) * np.sign(b)

        self._sc(start, half, d1)

        # g 操作
        left_bits = B[d1, start:mid]
        dst = L[d1, mid:end]
        np.subtract(1.0, 2.0 * left_bits, out=dst)
        dst *= a
        dst += b

        self._sc(mid, half, d1)

        # 合并 partial sums
        left = B[d1, start:mid]
        right = B[d1, mid:end]
        B[depth, start:mid] = left ^ right
        B[depth, mid:end] = right


# ---------------------------------------------------------------------------
# PolarDecoder 实例缓存 (避免重复初始化 + SSC 构建开销)
# ---------------------------------------------------------------------------
_decoder_cache: dict[tuple[int, int], PolarDecoder] = {}


def get_polar_decoder(n: int, k: int) -> PolarDecoder:
    """获取 PolarDecoder 实例, 相同 (N, K) 复用已创建的对象。"""
    key = (n, k)
    dec = _decoder_cache.get(key)
    if dec is None:
        dec = PolarDecoder(n, k)
        _decoder_cache[key] = dec
    return dec
