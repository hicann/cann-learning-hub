# 首轮候选池与排序

依据：semantic_cost_model.md、baseline_analysis.md、profiling/B0_initial_v2/summary.txt、actual generated C++、已安装rowwise/batched_short_reduction技能及仓内softmax.py显式Simd示例；历史性能不兼容且不继承。

当前B*=B0，全部比值1；总目标需至少5%耗时下降，等价1.05263倍加速；每项最大允许1.05*B0。精确各case阈值为aggregate.json各中位数乘1.05。

1. **H1 单VF行状态融合与跨chunk向量归约**（C1，全部九项）：让max/sum留寄存器，水平归约2N/64→2；exp生成时同时向量累加sum，减少e的一遍UB读4N；四VF→一VF，删除m/s UB往返。归一化保留逐元素除法，避免混入倒数精度变化。可证伪：生成代码须只两次水平归约、一次VF、相同GM；长N的VEC和各项最大AIV应下降。上界是原VEC+可删除固定段，不假定100%可消除；长N0.596us足以覆盖5%差距。风险：归约次序、S寄存器更新与store/load顺序，selected9+numeric30验证；API依仓内softmax.py simd_fold_block和实际tilelang/ascend/language/simd.py。成本一次编译九项+三轮采集；回退B0。
2. **H2 多行一次DMA配置族 R2/R4**：从当时B*独立构建，改变行tile/任务数，不改变每行算法。所有M被2/4整除。固定GM拷贝/调度M组→M/R，payload不变；理论只能消除固定段，不保证最大AIV或Task收益，并行减少风险高。特别针对N128/192的0.2–0.4us固定段；必须实测R>1。先R2全九项，再依据数据选R4代表或精确关闭。实际T.Kernel/T.copy/Simd行循环接口；成本九项编译+采集，回退当时B*。
3. **H3 行分母倒数复用**：在H1后依据VEC变化重排。N除法→1除法+N乘，最大可去掉原归一化除法成本；strict fp32无fastmath，数值边界必须验证。S.vdiv/vmul已核验。若H1已将热循环压低且目标达到则不优先执行。
4. **H4 persistent映射和stage配对**：仅M256有3.56波，候选需改成每核多行task循环并产生可重叠MTE/VEC生命周期；准入后不能因达到数值目标跳过自动/手工义务。当前无跨task内loop，预计收益仅可掩盖MTE/计算较小者，缺少当前结构生命周期证据，列末位待数据决定，不声称已实施。

队列外：全行寄存器保留e以去除UB写读，需要N2048最多32向量+中间state的实际寄存器/溢出证据；优先级暂低于H1复用。split-row跨核需要partial与额外launch，N<=2048整行语义不独立，不能当作inner_flatten。未提出语法等价改写。
