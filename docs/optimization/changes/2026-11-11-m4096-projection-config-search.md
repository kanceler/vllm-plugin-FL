# M4096四投影配置专项搜索

## 1. 基本信息与摘要

2026-11-11，C500。002采样确认16K prefill由M2048转为4096后，用户批准增加M4096四投影配置优化。新增 `benchmarks/diagnostics/search_metax_m4096.py`，源代码位于muxi-quantization工作树。隔离执行目录 `/data/m4096-config-search-20261111-01`，生产配置尚未修改。

## 2. 问题与原因分析

Gate/Up、QKV、O的既有专用MM配置只命中M2048，M4096走默认提供者。Down已有BM128区间候选，但仅有限搜索与复验，不能视为M4096专用全局最优。四组N/K不同，需要独立比较，避免直接复制M2048配置。

## 3. 具体实现

固定M4096，Gate/Up N12288/K2048、QKV N2560/K2048、O N2048/K2048使用原BF16 MM；Down N2048/K6144测INT8转换+MM+scale direct融合，转换BLOCK4096/warps4固定并计入完整调用，输出和临时内存分配也计时。BF16原始权重参考与Down量化参考分别验证。

每组BM/BN为64/128/256，BK32/64/128、warps4/8、stages2/4、basic/cpasync，216项有限候选，显式完整参数直接调用现有Kernel。保留三个BF16投影的M2048既有配置控制，Down保留原BM64与新BM128控制；失败不隐藏，非法访存立即停止。独立复验另保留实际注册MM/Down dispatch完整调用，不将预分配单Kernel计时当完整调用。

## 4. 实验与验证

初筛seed20261205，保留前六和合法控制；两新seed20261206/20261207，七轮随机交错完整调用、五轮随机交错图replay，预设L2<0.005。每组独立进程串行执行，启动前GPU空闲和9031检查；源码指纹、仓库差异、命令和前后资源状态归档。脚本AST检查通过，GPU结果待完成。#002在micro中不调度请求，M4096来自此前实际负载采样；后续组合测试才验证002运行期影响。

## 5. 与已有改动的关系

复用#002采样的相同最终量化overlay；只搜索配置，不改MM底层、转换、scale融合或M2048/小M既有策略。有稳定候选后还需实际提供者/dispatch接入复验，再更新精确M4096配置；不把旧精度报告自动视为新组合通过。

## 6. 合规性与风险

官方参数、数据和模型不变，无持久BF16副本或计时外转换。候选有限，不称全局最优。不得把默认配置收益与量化自身收益混淆；最终与同版本同配置提供者的BF16对照。未接入、未提交、未推送本轮变更。

## 7. 复现方式

```bash
PYTHONPATH=/data/m4096-config-search-20261111-01/overlay \
VLLM_FL_W8A16_SHAPE_DISPATCH=1 VLLM_FL_W8A16_FORCE_SEPARATE=0 \
python search_metax_m4096.py --projection gate_up --output <新目录>
```

其他projection为qkv/o/down，必须串行。实际命令在各组commands.json，候选screen.jsonl、selected.json及finalists.jsonl保留逐轮值；environment.json绑定MM和量化Kernel源码指纹。run_all.py遇失败停止后续任务。

### 搜索完成

四组各216项初筛，均137成功、79失败；两新seed及图重放复验已完成。Gate/Up赢家128/128/64/4/4/cpasync，完整调用1213.44/1212.48μs；QKV同配置298.05/298.77μs；O赢家128/128/128/4/4/cpasync，220.41/220.93μs。Down赢家仍为已有128/128/128/4/4/cpasync，564.33/564.43μs，实际dispatch564.57/564.43μs，无需重复更新Down配置。前三组尚须实际提供者接入复验；部分原配置控制项波动明显，不能直接沿用单seed最大加速比。原始归档 `/data/m4096-config-search-20261111-results.tgz`。

脚本复用了早期quant分组字段，因此原始BF16结果的kind字段也为quant；其projection及实际分支明确：Gate/Up、QKV、O使用BF16原始权重和原生MM，只有Down进行量化转换和融合。该字段不代表前三组被量化，原始记录保留，不追改已测源码。
