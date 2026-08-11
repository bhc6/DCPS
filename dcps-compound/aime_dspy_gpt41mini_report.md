# AIME-2025 × DSPy × GPT-4.1-mini 完整实验报告

**生成时间**: 2026-05-09  
**任务**: AIME-2025 整数答案数学题 (avg@5)  
**模型**: `openrouter/openai/gpt-4.1-mini`  
**框架**: DSPy (内部通过LiteLLM调用OpenRouter)  
**指标原则**: **仅报告 Top1 Test** (由验证集选出排名第一的prompt在test上的成绩)。**不使用 Best Test** (在test集上cherry-pick最高分prompt)，因为那构成test set leak，不是合法评估。

---

## 实验总览 (5 个日志, 2 个完成, 3 个失败)

| # | 日志 | 时间 | 脚本 | 状态 | Wandb run |
|---|------|------|------|------|-----------|
| 1 | `logs_baseline_dspy.txt` (126 KB) | 05-02 01:50 | `baseline_test_gpt41mini.py` (旧 prompt) | ✅ 完成 | qbbkcjhf |
| 2 | `logs_fewshot_dspy.txt` (1029 KB) | 05-02 04:25 | `dynamic_fewshot_gpt41mini.py` | ✅ 完成* | yo6weyl6 |
| 3 | `logs_aime_artifact_base_gpt41mini.txt` (5 KB) | 05-03 18:59 | `baseline_test_gpt41mini.py` (新版) | ❌ ValueError | gfk926js |
| 4 | `..._rerun.txt` (6 KB) | 05-04 01:41 | 同上 | ❌ TypeError | - |
| 5 | `..._rerun2.txt` (22 KB) | 05-04 02:19 | 同上 | ❌ 中断 (43/150) | 6xiy71qy |

*末尾因 GBK 编码 `UnicodeEncodeError` 在打印 emoji 时崩溃，但 FINAL REPORT 表格已落盘。

---

## 数据集配置 (所有实验共享)

```python
# 训练集 / 验证集
train_raw = load_dataset("AI-MO/aimo-validation-aime", "default", split="train")
random.Random(0).shuffle(train_split)
trainset = train_split[:45]   # 45 examples
valset = train_split[45:]     # 45 examples

# 测试集
test_raw = load_dataset("MathArena/aime_2025", "default", split="train")  # 30 题
testset = test_split * 5      # 复制5份 → 150 examples (avg@5)
```

| 项目 | 值 |
|------|-----|
| Train pool | 45 (来自 AIME 2022-2024) |
| Val pool | 45 (来自 AIME 2022-2024) |
| Test pool | 30 题 × 5 副本 = 150 (来自 AIME 2025) |
| Shuffle seed | `random.Random(0)` 仅用于shuffle，不影响全局PRNG |
| Metric | 整数精确匹配 (avg@5) |

---

## 实验 1: DSPy Baseline (旧 prompt) ✅

**日志**: `logs_baseline_dspy.txt`  
**Wandb**: `aime-math-baseline / qbbkcjhf` (2026-05-02 01:27 → 01:50)  
**脚本**: `baseline_test_gpt41mini.py` (2026-05-02 当时的版本)

### 实际使用的 prompt (从日志 line 20 确认)

```
Solve the following AIME problem. The final answer will always be an integer.
Show your work and reasoning in a clear, step-by-step manner, and conclude
your response with the final answer in the format: ### <answer>.
```

### 配置

| 参数 | 值 |
|------|-----|
| Solver Model | `openrouter/openai/gpt-4.1-mini` |
| Adapter | `dspy.ChatAdapter(use_json_adapter_fallback=False)` |
| Temperature | 1 |
| top_p | 0.95 |
| max_tokens | 16384 |
| API Key Env | `OPENROUTER_API_KEY_AIME_MATH_GPT41MINI` |
| Train / Val / Test | 45 / 45 / 150 |
| Metric | `artifact_aligned_AIME_integer_exact_match` |
| Program | `artifact_aligned_CoT_GenerateResponse` (DSPy ChainOfThought) |
| 评估时长 | 22:59 (并行8线程) |

### 结果

| 指标 | 值 |
|------|-----|
| **Top1 Test (avg@5)** | **47.33%** (71/150) |
| wandb summary | `test_score=0.4733` |

**⚠️ 注意**: 当前 `baseline_test_gpt41mini.py` 已改为使用 `artifact_default_instruction()` (即 `"Solve the problem and provide the answer in the correct format."`)，与该次实验所用 prompt **不同**。所以 47.33% 对应的是**历史代码版本**，不能直接复现当前脚本。

---

## 实验 2: DSPy Dynamic Few-Shot ✅

**日志**: `logs_fewshot_dspy.txt`  
**Wandb**: `aime-math-dynamic-fewshot / yo6weyl6` (2026-05-02 01:27 → 04:25)  
**脚本**: `dynamic_fewshot_gpt41mini.py`

### Base prompt (优化的起点)

```
You are a helpful assistant. You are given a question and you need to answer it.
The answer should be given at the end of your response in exactly the format
'### <final answer>'.
```

### 配置

| 参数 | 值 |
|------|-----|
| Solver Model | `openrouter/openai/gpt-4.1-mini` |
| Generator Model | `openrouter/openai/gpt-4.1-mini` |
| Temperature | 1 |
| top_p | 0.95 |
| max_tokens | 16384 |
| `NUM_ITERATIONS` | 20 |
| `NUM_FEWSHOT_EXAMPLES` | 3 |
| `TOP_K` | 5 |
| `VAL_SAMPLE_SIZE` | 15 (每轮 `random.sample(valset, 15)`，**v1风格**) |
| Train / Val / Test | 45 / 45 / 150 |
| Adapter | `dspy.ChatAdapter` |
| Wandb project | `aime-math-dynamic-fewshot` |

### 优化流程

```
for iter in 1..20:
    fewshot = random.sample(trainset, 3)
    prompt = generator_lm(metaprompt(fewshot, BASE_PROMPT))
    val_sample = random.sample(valset, 15)   # 每轮重新随机抽
    val_score = dspy.Evaluate(val_sample, prompt)
top5 = sort_by_val(all_results)[:5]
for prompt in top5:
    test_score = dspy.Evaluate(testset_150, prompt)
```

### 全部 20 轮验证分数 (从日志提取)

| Iter | Val | Iter | Val | Iter | Val | Iter | Val |
|------|-----|------|-----|------|-----|------|-----|
| 1 | 40.00% | 6 | 46.67% | 11 | 33.33% | 16 | 26.67% |
| 2 | 53.33% | 7 | 40.00% | 12 | 26.67% | 17 | 40.00% |
| 3 | 46.67% | 8 | 40.00% | **13** | **66.67%** | 18 | 53.33% |
| 4 | 60.00% | 9 | 60.00% | 14 | 53.33% | 19 | 40.00% |
| 5 | 40.00% | 10 | 60.00% | 15 | 26.67% | **20** | **66.67%** |

### Top 5 Final Report (按 val 降序)

| Rank | Iter | Val Score | Test Score (avg@5) |
|------|------|-----------|--------------------|
| **1** | **13** | **66.67%** | **44.67%** ← Top1 |
| 2 | 20 | 66.67% | 49.33% |
| 3 | 4 | 60.00% | 47.33% |
| 4 | 9 | 60.00% | 37.33% |
| 5 | 10 | 60.00% | 47.33% |

### 结果汇总 (仅以 Top1 为准)

| 指标 | 值 |
|------|-----|
| Top1 Val | 66.67% (Iter 13) |
| **Top1 Test (avg@5)** | **44.67%** |
| Top5 Avg Val | 62.67% |
| Top5 Avg Test | 45.20% |
| Val-Test Gap (Top1) | **+22.00%** (严重过拟合) |

**已知问题**:
- 训练期间多次出现 `LM response was truncated due to exceeding max_tokens=16384` 警告 (lines 3177, 5037, 7268, 10183, 11887, 13642) — GPT-4.1-mini 在AIME复杂题上推理可能截断
- 末尾因 `🏆` emoji 在 GBK 控制台触发 `UnicodeEncodeError` 而崩溃，wandb summary 的 best/avg 字段未上传 (但 FINAL REPORT 表格在崩溃前已打印)
- Top1 (Iter 13) 选中的 prompt 在 test 上**远不如** Top2 (Iter 20)，但 Top2 在 val 上并列第一，说明同 val 分数下选 Iter 13 是随机选择 (不可复现)

---

## 实验 3-5: Artifact Baseline (新版) ❌ 全部失败

**目标**: 用当前 `artifact_default_instruction()` (即 `"Solve the problem and provide the answer in the correct format."`) 重新评估 baseline。

### 实验 3 (gfk926js, 2026-05-03 18:59)

```
Traceback (most recent call last):
  File "examples/aime_math/artifact_aligned.py", line 104, in evaluate_on_dataset
    evaluator = dspy.Evaluate(
        devset=dataset,
        ...
        return_outputs=True,
    )
ValueError: `return_outputs` is no longer supported. Results are always
returned inside the `results` field of the `EvaluationResult` object.
```

**原因**: DSPy 升级后 `dspy.Evaluate` 移除了 `return_outputs` 参数。`artifact_aligned.py:104` 没有同步更新。

### 实验 4 (rerun, 2026-05-04 01:41)

```
TypeError: must be called with a dataclass type or instance
  at datasets/features/features.py:1467 in generate_from_dict
```

**原因**: `datasets` 库与 Python 3.13 的 dataclass 行为不兼容，`load_dataset("AI-MO/aimo-validation-aime")` 解析 features 时崩溃。

### 实验 5 (rerun2 / 6xiy71qy, 2026-05-04 02:19)

- 启动成功，跑到 **43/150 (29%)** 进度后日志中断
- 21.9 KB 日志末尾停在 `Average Metric: 24.00 / 43 (55.8%)`
- 没有 final test_score 落盘
- 中间速度 ~55.8%，但样本量太小不足以下结论

---

## 综合结果 (仅以 Top1 为准)

| 实验 | Prompt | Top1 Val | **Top1 Test** | Val-Test Gap | 状态 |
|------|--------|----------|---------------|--------------|------|
| Baseline (旧 prompt) | "Solve the following AIME..." (扩展) | - | **47.33%** | - | ✅ |
| Dynamic Few-Shot (DSPy) | (优化生成) | 66.67% | **44.67%** | +22.00% | ✅ |
| Baseline (artifact default) | "Solve the problem..." (短) | - | **未验证** | - | ❌ 全失败 |

---

## 跨框架对比 (仅以 Top1 为准)

| 实验 | 框架 | Prompt | **Top1 Test** |
|------|------|--------|---------------|
| Baseline (artifact default) | LiteLLM | "Solve the problem and provide..." | **49.33%** |
| Baseline (旧扩展 prompt) | DSPy | "Solve the following AIME..." | **47.33%** |
| Dynamic FS v1 (random val) | LiteLLM | (优化生成, 无base) | 46.00% |
| Dynamic FS (DSPy) | DSPy | (优化生成, 有BASE_PROMPT) | **44.67%** |
| Dynamic FS v2 (固定 val) | LiteLLM | (优化生成, 无base) | 43.33% |

---

## 核心发现

### ✅ 已验证的 DSPy 数字

1. **DSPy Baseline (旧 prompt)** = **47.33%** (logs_baseline_dspy.txt + wandb 0.4733)
2. **DSPy Dynamic Few-Shot Top1** = **44.67%** (Iter 13, Val 66.67%)

### ❌ 缺失/未验证

- **DSPy Baseline (当前 artifact-aligned prompt)**: 3 次重跑全部失败
- 历史 47.33% 用的是**不同 prompt**，**不能直接代表当前脚本的 baseline**

### ⚠️ 关键警告

1. **当前 `baseline_test_gpt41mini.py` 不可运行**
   - `dspy.Evaluate(return_outputs=True)` 已废弃
   - 需修改 `artifact_aligned.py:104` 移除该参数后重跑

2. **Prompt 漂移**
   - 旧 baseline (47.33%) 使用扩展 prompt + `### <answer>` 格式
   - 新 artifact-aligned prompt 是简短指令
   - 两者**直接可比性不强**

3. **DSPy 优化未能改善 baseline**
   - DSPy Dynamic FS Top1 = 44.67% < DSPy Baseline 47.33% (-2.66%)
   - 同样在 LiteLLM 上：v1=46.00%, v2=43.33% 都低于 LiteLLM Baseline 49.33%
   - **结论**: 在 GPT-4.1-mini + AIME-2025 任务上，dynamic few-shot prompt 优化**未能超越简单 baseline**

4. **max_tokens 截断**
   - 多次 `LM response truncated at max_tokens=16384` 警告
   - 可能影响复杂AIME题的推理完整性
   - 建议未来加大 `max_tokens` 或允许 num_retries

5. **未设全局 PRNG seed**
   - 重跑结果会变
   - Top1 是单次运行的快照，不代表期望性能

### 🔧 修复建议 (恢复 DSPy artifact baseline 验证)

```python
# examples/aime_math/artifact_aligned.py 第 104 行附近
evaluator = dspy.Evaluate(
    devset=dataset,
    metric=metric,
    num_threads=8,
    display_progress=True,
    failure_score=0.0,
    # return_outputs=True,  # ← 删除这一行 (已废弃)
)
result = evaluator(program)
# 改用: result.results 访问 per-example 结果
```

---

## 最终结论

| 项 | 值 |
|----|-----|
| **当前可信的 DSPy GPT-4.1-mini baseline** | **47.33%** (旧扩展 prompt, 非当前脚本) |
| **DSPy Dynamic Few-Shot Top1** | **44.67%** (低于 baseline 2.66%) |
| **DSPy artifact-default baseline** | **未验证** (代码与 dspy 新版不兼容) |
| **DSPy 优化在该任务上是否有效?** | **否** (Top1 反不如 baseline) |

---

## 附录: 文件路径

- DSPy 脚本:
  - `examples/aime_math/baseline_test_gpt41mini.py`
  - `examples/aime_math/dynamic_fewshot_gpt41mini.py`
  - `examples/aime_math/artifact_aligned.py` (问题: line 104 `return_outputs`)
- DSPy 日志:
  - `logs_baseline_dspy.txt` ✅
  - `logs_fewshot_dspy.txt` ✅
  - `logs_aime_artifact_base_gpt41mini.txt` ❌
  - `logs_aime_artifact_base_gpt41mini_rerun.txt` ❌
  - `logs_aime_artifact_base_gpt41mini_rerun2.txt` ❌
- Wandb 项目:
  - `awesome-prompt/aime-math-baseline`
  - `awesome-prompt/aime-math-dynamic-fewshot`
