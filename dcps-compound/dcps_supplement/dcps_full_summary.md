# DCPS 完整实验结果汇总

**DCPS** (Demonstration-Conditioned Prompt Search) — 2 个 benchmark × 2 个模型,共 4 个实验。

> **对齐说明 (Aug 2026):** LiveBench-Math 的测试分以**论文 Table 2(b)** 为准
> (qwen3-8b **65.08**、gpt-4.1-mini **59.52**,来自 20-iter server runs,见
> `../case_study/RECONCILE_PLUS_VS_PAPER.md`)。下方总表第 3–4 行的分数/迭代数/
> token/成本仍是被取代的 **60-iter 草稿**数据,**仅作 token/成本 provenance 保留**,
> 请勿作为论文数值引用。权威分数见本目录 `dcps_results.csv` / `dcps_results.md`。

测试分 = 验证集最高分对应 prompt 的**held-out 测试集**得分(公平 top-1:仅用验证集选 prompt,绝不用测试集选)。token/成本含断点续跑的累计。

## 总表(得分 + token + 成本)

| # | Benchmark | Model | 迭代 | 验证最高 | 测试分 | 泛化gap | 优化 tokens | 测试 tokens | 总 tokens | 优化 $ | 测试 $ | 总 $ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | HoVer | qwen3-8b | 7068 | 60.00% | 56.67% | 3.33 | 42.03M | 1.98M | 44.02M | 12.05 | 0.57 | 12.62 |
| 2 | HoVer | gpt-4.1-mini | 7068 | 56.67% | 54.67% | 2.00 | 31.21M | 1.35M | 32.56M | 23.20 | 0.96 | 24.16 |
| 3 | LiveBench-Math | gpt-4.1-mini | 1860 | 66.67% | 53.17% | 13.50 | 4.04M | 0.31M | 4.35M | 4.77 | 0.36 | 5.13 |
| 4 | LiveBench-Math | qwen3-8b | 1860 | 73.33% | 52.38% | 20.95 | 7.24M | 0.88M | 8.12M | 3.03 | 0.37 | 3.40 |
| | | | | | | **合计** | **84.52M** | **4.52M** | **89.05M** | **43.05** | **2.26** | **45.31** |

## Token 拆分(input / output)及调用数

| # | 实验 | 优化 input | 优化 output | 测试 input | 测试 output | API 调用数 |
|---|---|---:|---:|---:|---:|---:|
| 1 | HoVer qwen3-8b | 20.90M | 21.12M | 0.99M | 0.99M | 25,878 |
| 2 | HoVer gpt-4.1-mini | 22.28M | 8.93M | 1.00M | 0.35M | 26,795 |
| 3 | LiveBench-Math gpt-4.1-mini | 1.42M | 2.63M | 0.11M | 0.20M | 1,947 |
| 4 | LiveBench-Math qwen3-8b | 0.75M | 6.44M | 0.09M | 0.79M | 1,114 |

## 论文表格格式(测试分,0-100 百分比制)

DCPS 作为新方法行,接入 `generate_figures.ipynb` 的对比表(行=optimizer,列=(benchmark, program),值=测试分)。

### Model: qwen3-8b
| opt | (hoverBench, HoverMultiHop) | (LiveBenchMathBench, CoT) |
|-----|----------------------------:|--------------------------:|
| DCPS | 56.67 | 65.08 |

### Model: gpt-41-mini
| opt | (hoverBench, HoverMultiHop) | (LiveBenchMathBench, CoT) |
|-----|----------------------------:|--------------------------:|
| DCPS | 54.67 | 59.52 |

## 配置与数据集

| 项 | HoVer | LiveBench-Math |
|---|---|---|
| 迭代数 | 7068 | 1860 |
| 测试集样本数 | 300 | 126 |
| few-shot 数 | 3 | 3 |
| top_k(报告) | 1 | 1 |
| 最佳 prompt 迭代 | — | gpt: 18 / qwen: 42 |

采样参数:temperature 0.6,top_p 0.95,top_k 20,max_tokens 8192,num_threads 16。推理经 OpenRouter(`openrouter/qwen/qwen3-8b`、`openrouter/openai/gpt-4.1-mini`)。

## 说明

- **成本结构**:HoVer(228 轮 × 测试集 300)比 LiveBench-Math(60 轮 × 测试集 126)贵约一个量级。最贵是 HoVer gpt-4.1-mini($24.16)。
- **qwen3-8b 是 thinking 模型**:output tokens 远超 input(如 LiveBench 优化 0.75M in / 6.44M out),但单价低,总成本反而低于 gpt。
- **测试成本占比小**(每个 <$1),主要花在优化阶段的验证评估。
- **泛化 gap**:验证最高分是 cherry-picked(多轮迭代在固定验证集上取 max,有选择偏差),故高于测试分,对应论文的 "generalization gap" 概念。LiveBench qwen 的 gap 最大(20.95)。
- **LiveBench qwen3-8b**:论文 Table 2(b) 权威测试分 = **65.08%**(20-iter server run,见 `../case_study/RECONCILE_PLUS_VS_PAPER.md`)。本表第 4 行的 52.38% 是被取代的 60-iter 草稿(W&B run `07tj4oz7`,66/126;草稿内另一次重跑因 OpenRouter `SSL: UNEXPECTED_EOF` 网络错误样本更多各判 0 + temp 0.6 方差得 38.10%,故草稿以干净的 52.38% 为准),此处仅作 token/成本 provenance 保留。表中 token/成本取日志最近一次 run 块(优化 $3.03 与 wandb 一致)。
