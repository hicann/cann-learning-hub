# Ascend C框架插入流水同步的Add算子样例

## 概述

本样例基于Ascend C Python工程，介绍了通过Ascend C框架插入流水同步的Add算子实现。

## 支持的AI处理器
- Ascend 910C
- Ascend 910B

## 样例实现

- 样例规格

  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">Add</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">\</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">\</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">\</td><td align="center">float</td><td align="center">ND</td></tr>
  </table>

- 算子实现

  本样例实现的是[8, 2048]的Add算子，计算过程中的流水同步通过Ascend C框架实现。
  - kernel实现

    Add算子的实现流程：首先将Global Memory上的输入Tensor x_gm和y_gm搬运到Local Memory，分别存储在x_local、y_local，然后对x_local、y_local执行加法操作，计算结果存储在z_local中，最后将输出数据从z_local搬运至Global Memory上的输出Tensor z_gm中。

    计算过程中用到的Local Memory x_local、y_local、z_local通过TPipe alloc_tensor接口获取，free_tensor接口释放，通过enque/deque 接口传递，在此过程中，Ascend C框架会自动插入对应的同步事件，无需调用set_flag/wait_flag设置同步。
  - tiling实现

    本示例算子使用了2个tiling参数：total_length、TILE_NUM。total_length是指需要计算的数据量大小，TILE_NUM是指每个核上总计算数据分块个数。比如，total_length这个参数传递到kernel侧后，可以通过除以参与计算的核数，得到每个核上的计算量，这样就完成了多核数据的切分。

## 编译执行
环境配置请参考[quick_start.md](../../../docs/quick_start.md#envready)。完成环境配置后，执行如下命令可进行功能验证。
```
cd pyasc/python/tutorials/02_add_framework
python3 add_framework.py -r [RUN_MODE] -v [SOC_VERSION]
```
其中脚本参数说明如下：
- RUN_MODE：编译执行方式，可选择NPU仿真，NPU上板，对应参数分别为[Model/NPU]。
- SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。

示例如下，Ascendxxxyy请替换为实际的AI处理器型号。
```
python3 add_framework.py -r Model -v Ascendxxxyy
```
用例执行完成，打屏信息出现“Sample add_framework run success.”，说明样例执行成功。

注：torch_npu尚不支持Python 3.12，因此该用例在Python 3.12环境上无法执行NPU模式。