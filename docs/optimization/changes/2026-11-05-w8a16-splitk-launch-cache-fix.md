# Split-K 显式配置启动缓存修复

## 1. 基本信息与摘要

- 日期：2026-11-05；设备：沐曦 C500。
- 文件：FlagGems `src/flag_gems/runtime/backend/_metax/ops/weight_int8pack_mm.py`；诊断脚本 `benchmarks/diagnostics/verify_splitk_launch_options.py`。
- 目标：使显式 Split-K 候选的 `num_warps`、`num_stages` 实际传给编译器，避免搜索标签与执行 Kernel 不一致。

## 2. 问题与原因分析

原 `_configured_splitk` 通过共用的 `@libentry()` 对象启动 partial Kernel。归档版本的 `LibEntry.key()` 使用 JIT 参数构造缓存键；tile 与 S 是 constexpr，但 `num_warps`、`num_stages` 是启动选项，未进入键。同 tile/S 的后续不同启动参数可能复用先前编译的 Kernel。原始搜索计时依然是真实调用，但不能认证标注的完整配置。详细归档见 `2026-11-05-w8a16-completed-sweeps-analysis.md`。

## 3. 具体实现

仅将 `_configured_splitk` 的 partial 调用改为 `weight_int8pack_mm_splitk_partial.jit_function.run(...)`，显式传入原有 grid、pipeline、warps、stages 和 `warmup=False`。Triton JIT 按完整编译选项区分产物。reduce/scale、workspace、数值流程、算法分支均未改变。Dense 仍待其容器有资源时单独处理；本次不修改全局 LibEntry。

## 4. 实验与验证

独立新目录 `/root/experiments/2026-11-05-w8a16-splitk-launch-fix-01/`，从旧实验副本创建，旧目录保持原样。首次验证因脚本遗漏 Triton `warmup` 必填参数而失败，修正后第二次退出码 0。固定 M3/N2048/K2048、BM16/BN64/BK128/S2，顺序请求 warps/stages 为 `(2,1)→(8,3)→(2,1)`；读取真实 `CompiledKernel.metadata` 分别为 `(2,1)→(8,3)→(2,1)`。前两次是不同 Kernel 对象，第三次复用第一次。修复后完整 Split-K 调用对 FP32 参考的相对 L2 为 `2.1190828647377202e-06`。命令输出在 `launch-test-r2.log`，退出码在 `launch-test-r2-exit.txt`。这验证配置切换和一个尾块 Shape；完整候选搜索、性能收益及模型精度均待复测。

CPU 搜索策略测试 12 项通过；源码语法解析和 FlagGems diff 检查通过。GPU 证据已下载至本机 `experiment-artifacts/2026-11-05-w8a16-splitk-wide-01/launch-fix-evidence.tar.gz`。GPU 空闲核对后，已在上述远端新目录启动完整 Split-K 搜索，控制进程启动 PID 1246；启动时 `results/controller.lock` 与 preflight 目录已生成。搜索仍在运行，结果与收益待验证。

## 5. 与已有改动的关系

修复作用于既有 W8A16 Split-K 显式配置入口，直接纠正 11 月 5 日归档审计发现的启动参数缓存别名。原始 60 点结果保留为线索，旧赢家仍不能直接固化。

## 6. 合规性与风险

不修改官方命令、评测脚本、模型行为或原生 BF16 GEMM。诊断为独立算子测试，不代表官方吞吐收益。直接使用 Triton JIT 的编译缓存，需要在后续多 Shape 重测中继续检查编译、资源和数值；当前仅验证一组 Shape/两套启动参数。

## 7. 复现方式

在 C500 的 Conda/MACA 环境中，将此 FlagGems 源码放入 `PYTHONPATH` 优先位置，并执行：

```bash
python benchmarks/diagnostics/verify_splitk_launch_options.py
```

脚本固定两套启动参数，核对 `CompiledKernel.metadata`、编译产物对象和完整调用相对误差；失败时非零退出。远端新目录及其日志包含本次实际执行证据。下一步在相同实例以新的结果目录重测代表 tile/S 的 warps/stages，再做 BF16、Split-K、Separate 同机完整调用对照。
