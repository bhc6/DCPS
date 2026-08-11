# 原始实验 Cost 与 Token 报告

## 1. 报告口径

本报告 **全部使用实验日志中的原始结果**，不再引入任何基于价格表的重新估算或新算法。

原始数据来源：

```text
seed_0/<run_dir>/evaluation_results/evaluation_result.txt
```

文件字段：

```text
score,cost,input_tokens,output_tokens,optimizer,optimizer_cost,optimizer_input_tokens,optimizer_output_tokens
```

其中：

- **Test cost/input/output**：最终测试阶段的原始日志结果。
- **Optimizer cost/input/output**：优化阶段的原始日志结果。
- **Total cost** = `cost + optimizer_cost`。
- **Total input** = `input_tokens + optimizer_input_tokens`。
- **Total output** = `output_tokens + optimizer_output_tokens`。
- **Total tokens** = `Total input + Total output`。

Baseline 不运行 optimizer，因此 optimizer cost/token 为 0。

## 2. 原始 cost/token 的代码来源

实验代码通过累加 LM history 中的原始记录得到 cost 和 token：

```python
def calculate_stats(lm) -> tuple[float, int, int]:
    cost = 0
    input_tokens = 0
    output_tokens = 0
    for i, trace in enumerate(lm.history):
        cost += trace.get("cost", None) or 0
        input_tokens += trace.get("usage", 0).get("prompt_tokens", 0)
        output_tokens += trace.get("usage", 0).get("completion_tokens", 0)

    return cost, input_tokens, output_tokens
```

因此本报告中的 cost 和 token 均为运行时日志记录值。

## 3. 完整原始结果表

| Benchmark | Model | Method | Score | Test cost | Test in | Test out | Opt cost | Opt in | Opt out | Total cost | Total in | Total out | Total tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AIMEBench_CoT | gpt-41-mini | Baseline | 49.33 | 0.9243656 | 24,399 | 347,277 | 0.0000000 | 0 | 0 | 0.9243656 | 24,399 | 347,277 | 371,676 |
| AIMEBench_CoT | gpt-41-mini | GEPA | 59.33 | 1.1226996 | 119,886 | 391,026 | 18.1436428 | 4,518,043 | 9,923,246 | 19.2663424 | 4,637,929 | 10,314,272 | 14,952,201 |
| AIMEBench_CoT | gpt-41-mini | MIPROv2-Heavy | 51.33 | 0.8055888 | 112,940 | 293,757 | 11.3603816 | 5,795,259 | 5,157,799 | 12.1659704 | 5,908,199 | 5,451,556 | 11,359,755 |
| HotpotQABench_HotpotMultiHop | gpt-41-mini | Baseline | 38.00 | 0.5700492 | 767,451 | 164,418 | 0.0000000 | 0 | 0 | 0.5700492 | 767,451 | 164,418 | 931,869 |
| HotpotQABench_HotpotMultiHop | gpt-41-mini | GEPA | 69.00 | 0.8663164 | 1,477,051 | 227,865 | 20.7346100 | 21,433,903 | 3,305,995 | 21.6009264 | 22,910,954 | 3,533,860 | 26,444,814 |
| HotpotQABench_HotpotMultiHop | gpt-41-mini | GEPA-MERGE | 65.67 | 0.8763416 | 1,476,758 | 233,628 | 19.1344516 | 13,801,004 | 1,675,390 | 20.0107932 | 15,277,762 | 1,909,018 | 17,186,780 |
| HotpotQABench_HotpotMultiHop | gpt-41-mini | MIPROv2-Heavy | 58.00 | 0.8694892 | 2,006,275 | 161,814 | 19.1923092 | 39,872,835 | 3,254,327 | 20.0617984 | 41,879,110 | 3,416,141 | 45,295,251 |
| HotpotQABench_HotpotMultiHop | qwen3-8b | Baseline | 42.33 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0 |
| HotpotQABench_HotpotMultiHop | qwen3-8b | GEPA | 62.33 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0 |
| HotpotQABench_HotpotMultiHop | qwen3-8b | GEPA-MERGE | 64.33 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0 |
| HotpotQABench_HotpotMultiHop | qwen3-8b | MIPROv2-Heavy | 55.33 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0 |
| hoverBench_HoverMultiHop | gpt-41-mini | Baseline | 46.33 | 0.7225640 | 832,974 | 243,359 | 0.0000000 | 0 | 0 | 0.7225640 | 832,974 | 243,359 | 1,076,333 |
| hoverBench_HoverMultiHop | gpt-41-mini | GEPA | 51.67 | 1.0707156 | 1,606,613 | 329,104 | 23.7677768 | 21,060,650 | 4,836,444 | 24.8384924 | 22,667,263 | 5,165,548 | 27,832,811 |
| hoverBench_HoverMultiHop | gpt-41-mini | GEPA-MERGE | 56.67 | 1.0516484 | 1,039,401 | 236,948 | 24.5222000 | 16,892,100 | 4,282,283 | 25.5738484 | 17,931,501 | 4,519,231 | 22,450,732 |
| hoverBench_HoverMultiHop | gpt-41-mini | MIPROv2-Heavy | 48.33 | 0.9834676 | 2,205,501 | 277,564 | 22.7952836 | 43,822,464 | 5,087,521 | 23.7787512 | 46,027,965 | 5,365,085 | 51,393,050 |
| hoverBench_HoverMultiHop | qwen3-8b | Baseline | 35.33 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0 |
| hoverBench_HoverMultiHop | qwen3-8b | GEPA | 52.33 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0 |
| hoverBench_HoverMultiHop | qwen3-8b | GEPA-MERGE | 51.67 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0 |
| hoverBench_HoverMultiHop | qwen3-8b | MIPROv2-Heavy | 47.33 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0 |
| IFBench_IFBenchCoT2StageProgram | gpt-41-mini | Baseline | 47.79 | 0.5133340 | 72,409 | 90,092 | 0.0000000 | 0 | 0 | 0.5133340 | 72,409 | 90,092 | 162,501 |
| IFBench_IFBenchCoT2StageProgram | gpt-41-mini | GEPA | 52.72 | 0.6900840 | 599,910 | 281,325 | 7.7160720 | 5,372,392 | 2,237,085 | 8.4061560 | 5,972,302 | 2,518,410 | 8,490,712 |
| IFBench_IFBenchCoT2StageProgram | gpt-41-mini | GEPA-MERGE | 55.95 | 0.6614608 | 589,056 | 267,109 | 7.7385624 | 4,890,476 | 2,030,398 | 8.4000232 | 5,479,532 | 2,297,507 | 7,777,039 |
| IFBench_IFBenchCoT2StageProgram | gpt-41-mini | MIPROv2-Heavy | 49.15 | 0.7036384 | 959,936 | 330,206 | 7.2420348 | 9,961,526 | 2,613,496 | 7.9456732 | 10,921,462 | 2,943,702 | 13,865,164 |
| IFBench_IFBenchCoT2StageProgram | qwen3-8b | Baseline | 36.90 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0 |
| IFBench_IFBenchCoT2StageProgram | qwen3-8b | GEPA | 38.61 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0 |
| IFBench_IFBenchCoT2StageProgram | qwen3-8b | GEPA-MERGE | 28.23 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0 |
| IFBench_IFBenchCoT2StageProgram | qwen3-8b | MIPROv2-Heavy | 36.22 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0 |
| LiveBenchMathBench_CoT | gpt-41-mini | Baseline | 58.20 | 0.3506908 | 59,566 | 189,570 | 0.0000000 | 0 | 0 | 0.3506908 | 59,566 | 189,570 | 249,136 |
| LiveBenchMathBench_CoT | gpt-41-mini | GEPA | 59.43 | 0.4121828 | 195,937 | 232,390 | 5.2235256 | 2,729,962 | 2,591,433 | 5.6357084 | 2,925,899 | 2,823,823 | 5,749,722 |
| LiveBenchMathBench_CoT | gpt-41-mini | GEPA-MERGE | 64.13 | 0.4214364 | 143,647 | 227,966 | 5.6199192 | 2,358,424 | 2,771,103 | 6.0413556 | 2,502,071 | 2,999,069 | 5,501,140 |
| LiveBenchMathBench_CoT | gpt-41-mini | MIPROv2-Heavy | 61.84 | 0.3718724 | 426,517 | 192,847 | 5.0929772 | 7,196,344 | 1,878,195 | 5.4648496 | 7,622,861 | 2,071,042 | 9,693,903 |
| Papillon_PAPILLON | gpt-41-mini | Baseline | 78.57 | 0.3627144 | 303,519 | 132,623 | 0.0000000 | 0 | 0 | 0.3627144 | 303,519 | 132,623 | 436,142 |
| Papillon_PAPILLON | gpt-41-mini | GEPA | 94.47 | 0.4188188 | 434,568 | 120,082 | 6.4211864 | 6,236,890 | 2,071,522 | 6.8400052 | 6,671,458 | 2,191,604 | 8,863,062 |
| Papillon_PAPILLON | gpt-41-mini | GEPA-MERGE | 96.46 | 0.4088496 | 452,667 | 117,489 | 6.3621920 | 6,590,681 | 2,054,873 | 6.7710416 | 7,043,348 | 2,172,362 | 9,215,710 |
| Papillon_PAPILLON | gpt-41-mini | MIPROv2-Heavy | 83.37 | 0.4407056 | 769,562 | 128,351 | 5.5852656 | 10,101,491 | 1,361,542 | 6.0259712 | 10,871,053 | 1,489,893 | 12,360,946 |
| Papillon_PAPILLON | qwen3-8b | Baseline | 80.82 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0 |
| Papillon_PAPILLON | qwen3-8b | GEPA | 91.85 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0 |
| Papillon_PAPILLON | qwen3-8b | GEPA-MERGE | 86.26 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0 |
| Papillon_PAPILLON | qwen3-8b | MIPROv2-Heavy | 81.55 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0.0000000 | 0 | 0 | 0 |

## 4. 按 Model/Method 汇总

| Model | Method | Runs | Test cost | Optimizer cost | Total cost | Average cost | Total input | Total output | Total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-41-mini | Baseline | 6 | 3.4437180 | 0.0000000 | 3.4437180 | 0.5739530 | 2,060,318 | 1,167,339 | 3,227,657 |
| gpt-41-mini | GEPA | 6 | 4.5808172 | 82.0068136 | 86.5876308 | 14.4312718 | 65,785,805 | 26,547,517 | 92,333,322 |
| gpt-41-mini | GEPA-MERGE | 5 | 3.4197368 | 63.3773252 | 66.7970620 | 13.3594124 | 48,234,214 | 13,897,187 | 62,131,401 |
| gpt-41-mini | MIPROv2-Heavy | 6 | 4.1747620 | 71.2682520 | 75.4430140 | 12.5738357 | 123,230,650 | 20,737,419 | 143,968,069 |
| qwen3-8b | Baseline | 4 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0 | 0 | 0 |
| qwen3-8b | GEPA | 4 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0 | 0 | 0 |
| qwen3-8b | GEPA-MERGE | 4 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0 | 0 | 0 |
| qwen3-8b | MIPROv2-Heavy | 4 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0 | 0 | 0 |

## 5. GPT 方法平均花费排名

按原始 `Total cost / Runs` 从低到高：

| Rank | Method | Runs | Total cost | Average cost |
|---:|---|---:|---:|---:|
| 1 | Baseline | 6 | 3.4437180 | 0.5739530 |
| 2 | MIPROv2-Heavy | 6 | 75.4430140 | 12.5738357 |
| 3 | GEPA-MERGE | 5 | 66.7970620 | 13.3594124 |
| 4 | GEPA | 6 | 86.5876308 | 14.4312718 |

## 6. 结论

- 本报告 **仅使用原始日志结果**，不做任何价格表重算。
- GPT 的 cost/token 包含 **optimizer 阶段 + test 阶段**。
- Baseline 没有 optimizer 阶段，因此 optimizer cost/token 为 0。
- Qwen3 的原始日志中 cost/token 均为 0；本报告保留原始值，不做估算。
- 在 GPT 优化方法中，按平均原始 cost：
  - **MIPROv2-Heavy** 最低：`12.5738357`
  - **GEPA-MERGE** 次之：`13.3594124`
  - **GEPA** 最高：`14.4312718`
