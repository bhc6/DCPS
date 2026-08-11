# AIME-2025 LiteLLM GPT-4.1-mini: v1 vs v2 完整对比报告

**生成时间**: 2026-05-08  
**对比范围**: AIME-2025 LiteLLM (纯LiteLLM，不经DSPy) + GPT-4.1-mini  
**脚本**: `dynamic_fewshot_litellm_gpt41mini.py` (v1) vs `dynamic_fewshot_litellm_v2_gpt41mini.py` (v2)

---

## 目录

1. [核心区别 TL;DR](#核心区别-tldr)
2. [数据集配置](#数据集配置)
3. [模型与采样配置](#模型与采样配置)
4. [优化流程配置](#优化流程配置)
5. [v1 vs v2 代码差异](#v1-vs-v2-代码差异)
6. [完整结果对比](#完整结果对比)
7. [Wandb元数据](#wandb元数据)
8. [优缺点分析](#优缺点分析)

---

## 核心区别 TL;DR

**唯一本质差异**: 验证集的采样策略

| 维度 | v1 | v2 |
|------|-----|-----|
| 验证集采样 | **每轮随机重采样** | **固定头切片** |
| 验证集来源 | `random.sample(valset, 15)` 每轮不同 | `valset[:15]` 所有轮次相同 |
| 验证集大小 | 15 (从45 valset随机抽样) | 15 (从45 valset头15个) |
| 评估公平性 | 低 (每个prompt看到不同问题) | 高 (所有prompt看到相同问题) |
| 选择噪声 | 高 | 低 |

**其他所有配置完全相同**，包括数据集、模型、温度、迭代次数、fewshot数、top-K等。

---

## 数据集配置

### 训练集 / 验证集 (共同)

**源**: `AI-MO/aimo-validation-aime` (HuggingFace)

```python
train_raw = load_dataset("AI-MO/aimo-validation-aime", "default", split="train")
# 90 examples total (AIME 2022-2024)

random.Random(0).shuffle(train_split)  # seed=0 shuffle

n = len(train_split)  # n = 90
trainset = train_split[: n // 2]   # 45 examples
valset = train_split[n // 2 :]     # 45 examples
```

| 项目 | v1 | v2 |
|------|-----|-----|
| Full train pool | 45 | 45 |
| Full val pool | 45 | 45 |
| Shuffle seed | 0 | 0 |
| Split比例 | 50/50 | 50/50 |

### 测试集 (共同)

**源**: `MathArena/aime_2025` (HuggingFace)

```python
test_raw = load_dataset("MathArena/aime_2025", "default", split="train")
# 30 examples (AIME 2025)
testset = test_split * 5  # 复制5份 → 150 examples (avg@5)
```

| 项目 | v1 | v2 |
|------|-----|-----|
| Raw test size | 30 | 30 |
| Replication | 5x | 5x |
| Final test size | 150 | 150 |
| Metric | avg@5 精确整数匹配 | avg@5 精确整数匹配 |

### 每轮验证子集 (关键差异)

```python
# v1 (dynamic_fewshot_litellm_gpt41mini.py, line 281)
val_sample = random.sample(valset, min(VAL_SAMPLE_SIZE, len(valset)))  # 每轮抽15
val_score = evaluate_on_dataset(generated_prompt, val_sample, ...)
```

```python
# v2 (dynamic_fewshot_litellm_v2_gpt41mini.py, line 264)
valset = full_valset[:VAL_SAMPLE_SIZE]  # main开头一次性确定，固定不变
# 循环内不再重新采样
val_score = evaluate_on_dataset(generated_prompt, valset, ...)
```

| 指标 | v1 | v2 |
|------|-----|-----|
| `VAL_SAMPLE_SIZE` | 15 | 15 |
| 采样方式 | `random.sample(valset, 15)` | `valset[:15]` |
| 每轮验证集是否相同 | ❌ 否 | ✅ 是 |
| 选择偏差 | 高（候选之间不可比） | 无 |

---

## 模型与采样配置

### 求解器 (Solver) LLM

| 参数 | v1 | v2 |
|------|-----|-----|
| Model | `openrouter/openai/gpt-4.1-mini` | `openrouter/openai/gpt-4.1-mini` |
| Temperature | 1 (int, 即1.0) | 1 (int, 即1.0) |
| top_p | 0.95 | 0.95 |
| max_tokens | 16384 | 16384 |
| top_k | (未设置) | (未设置) |
| Provider Pin | None | None |
| num_retries | 默认 | 默认 |

### Generator LLM (生成新prompt)

| 参数 | v1 | v2 |
|------|-----|-----|
| Model | `openrouter/openai/gpt-4.1-mini` | `openrouter/openai/gpt-4.1-mini` |
| 采样参数 | 同solver (共用llm_kwargs) | 同solver (共用llm_kwargs) |

**注**: 与Qwen3-8B变体不同，GPT-4.1-mini两个脚本**不pin provider**、**不设top_k**，对齐`gepa-artifact/run_aime_api.sh`中gpt-4.1-mini采样协议。

### API 配置

| 项目 | v1 | v2 |
|------|-----|-----|
| API Key Env (primary) | `OPENROUTER_API_KEY_AIME_MATH_GPT41MINI` | `OPENROUTER_API_KEY_AIME_MATH_V2` |
| API Key Env (fallback) | `OPENROUTER_API_KEY` | `OPENROUTER_API_KEY` |
| API Base | `OPENROUTER_API_BASE` (env) | `OPENROUTER_API_BASE` (env) |

**注意**: v1 和 v2 使用 **不同的 API Key 环境变量** (`_GPT41MINI` vs `_V2`)。实际实验时两个都fallback到`OPENROUTER_API_KEY`。

---

## 优化流程配置

| 参数 | v1 | v2 |
|------|-----|-----|
| `NUM_ITERATIONS` | **20** | **20** |
| `NUM_FEWSHOT_EXAMPLES` | **3** | **3** |
| `TOP_K` | **5** | **5** |
| `VAL_SAMPLE_SIZE` | **15** | **15** |
| `NUM_THREADS` | **8** | **8** |
| Base prompt | (none) | (none) |

### 流程 (完全相同的骨架)

1. 从`trainset` (45样本) 随机抽取 3 条 few-shot 示例
2. 用metaprompt让generator LM生成新system prompt
3. **[关键差异]** 在验证集上评估这个prompt
   - **v1**: `random.sample(valset, 15)` → 每轮重新采样
   - **v2**: `valset[:15]` → 固定头切片
4. 记录 `(prompt, val_score)` 到 `all_results`
5. 迭代20次
6. 按val_score降序排序，取Top 5
7. 每个Top-5 prompt在**完整150样本test set** (avg@5)上评估
8. 报告Top1、Best、Top5平均

### Few-shot sampling函数 (两版本相同)

```python
def sample_fewshot_examples(trainset, num_examples=3):
    sampled = random.sample(trainset, min(num_examples, len(trainset)))
    parts = []
    for i, ex in enumerate(sampled, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Problem: {ex.problem}")
        if ex.solution:
            parts.append(f"Solution: {ex.solution}")
        parts.append(f"Answer: {ex.answer}\n")
    return "\n".join(parts).strip()
```

### Metaprompt (两版本相同)

```
You are an expert prompt engineer for AI systems.

Based on the few-shot examples below, design an effective prompt that 
will guide an AI to solve math problems accurately.

Here are the few-shot examples to analyze:

{fewshot_examples}

Now, generate a prompt that incorporates insights from these examples:
```

### Answer extraction (两版本相同)

1. 优先: `###\s*(\d+)` 格式
2. 其次: `\boxed{...}` (去LaTeX装饰)
3. 兜底: 文本中最后一个整数

---

## v1 vs v2 代码差异

### 完整差异清单

| # | 维度 | v1 | v2 | 影响 |
|---|------|-----|-----|------|
| 1 | 验证集采样 | `random.sample(valset, 15)` 每轮 | `valset[:15]` 一次 | **核心算法差异** |
| 2 | unpack变量名 | `trainset, valset, testset` | `trainset, full_valset, testset` | 命名 |
| 3 | 每轮PRNG调用次数 | 2次 (trainset + valset) | 1次 (trainset) | **PRNG状态发散** |
| 4 | docstring | 简短 | 详细对比v1 | 文档 |
| 5 | wandb run name | `litellm_gpt41mini_*` | `litellm_v2_gpt41mini_*` | 标识 |
| 6 | wandb config | 11字段 | 16字段 (`version`/`val_strategy`等) | 元数据 |
| 7 | CLI输出 | 无固定val提示 | `[OK] Fixed validation pool: 15` | 提示 |

### 完全相同的部分

- ✅ `MathExample` dataclass
- ✅ `load_math_dataset()` (含seed=0 shuffle、50/50 split、testset×5)
- ✅ `extract_answer()` (### → \boxed → 末尾整数)
- ✅ `call_llm()` (litellm.completion封装)
- ✅ `evaluate_single()` (返回 (score, feedback))
- ✅ `evaluate_on_dataset()` (ThreadPoolExecutor、tqdm进度条)
- ✅ `sample_fewshot_examples()` (`random.sample(trainset, 3)`格式化)
- ✅ `create_metaprompt()` (字面相同的英文模板)
- ✅ `generate_prompt_with_llm()` (失败时返回空串)
- ✅ `llm_kwargs` (temperature=1, top_p=0.95, max_tokens=16384)
- ✅ Test phase代码 (Top5每个prompt在150样本上avg@5)
- ✅ wandb metric定义
- ✅ Final Report格式与wandb.summary写法

### 关键代码diff (集中在main()的验证评估部分)

```python
# ======== v1 ========
def main():
    # ...
    trainset, valset, testset = load_math_dataset()
    # valset保持45条，不裁剪

    for iteration in range(NUM_ITERATIONS):
        fewshot_examples = sample_fewshot_examples(trainset, NUM_FEWSHOT_EXAMPLES)
        metaprompt = create_metaprompt(fewshot_examples)
        generated_prompt = generate_prompt_with_llm(metaprompt, ...)

        # ★ 每轮随机采样15个
        val_sample = random.sample(valset, min(VAL_SAMPLE_SIZE, len(valset)))
        val_score = evaluate_on_dataset(generated_prompt, val_sample, ...)
```

```python
# ======== v2 ========
def main():
    # ...
    trainset, full_valset, testset = load_math_dataset()

    # ★ 一次性确定fixed validation pool
    valset = full_valset[:VAL_SAMPLE_SIZE]  # valset[:15]

    for iteration in range(NUM_ITERATIONS):
        fewshot_examples = sample_fewshot_examples(trainset, NUM_FEWSHOT_EXAMPLES)
        metaprompt = create_metaprompt(fewshot_examples)
        generated_prompt = generate_prompt_with_llm(metaprompt, ...)

        # ★ 所有iteration共用同一个valset (no resample)
        val_score = evaluate_on_dataset(generated_prompt, valset, ...)
```

### 其他微小差异

| 项目 | v1 | v2 | 说明 |
|------|-----|-----|------|
| Docstring | 简短 | 详细描述v1-v2差异 | 文档层面 |
| wandb run name | `litellm_gpt41mini_{N}iter_{K}shot` | `litellm_v2_gpt41mini_{N}iter_{K}shot` | 命名区分 |
| wandb config字段 | 无版本标记 | 多出`version=v2-fixed-val`、`val_strategy`等字段 | 元数据 |
| print提示 | 无"fixed val"标注 | 有"[OK] Fixed validation pool"提示 | CLI输出 |
| 返回值unpack | `trainset, valset, testset` | `trainset, full_valset, testset` | 变量命名 |

---

## 完整结果对比

### Baseline (参考)

| 实验 | Val | Test (avg@5) |
|------|-----|--------------|
| Baseline (artifact默认指令) | 42.22% | **49.33%** |

Baseline prompt: `Solve the problem and provide the answer in the correct format.`

### v1 结果 (logs_fewshot_litellm.txt:2529-2533)

**Top 5 by validation** (按val降序):

| Rank | Iter | Val Score | Test Score (avg@5) |
|------|------|-----------|--------------------|
| 1 | 13 | 73.33% | **46.00%** ← Top1 |
| 2 | 11 | 60.00% | 42.00% |
| 3 | 2  | 53.33% | 45.33% |
| 4 | 5  | 53.33% | 47.33% |
| 5 | 9  | 53.33% | 32.67% |

**注**: v1日志在Final Report打印后被截断，**未保留 wandb summary**。

**汇总** (仅以 Top1 为准):
- Top1 Val: **73.33%** (Iter 13)
- **Top1 Test: 46.00%**
- Top5 Avg Val: 58.67%
- Top5 Avg Test: 42.67%
- Val-Test Gap (Top1): **+27.33%** (严重过拟合)

### v2 结果 (logs_v2_gpt41mini.txt:2535-2541)

**Top 5 by validation** (按val降序):

| Rank | Iter | Val Score | Test Score (avg@5) |
|------|------|-----------|--------------------|
| 1 | 1  | 60.00% | **43.33%** ← Top1 |
| 2 | 19 | 60.00% | 40.67% |
| 3 | 2  | 53.33% | 49.33% |
| 4 | 11 | 53.33% | 42.00% |
| 5 | 12 | 53.33% | 42.00% |

**wandb summary** (验证):
```
avg_test_score_top_k = 0.43467
avg_val_score_top_k  = 0.56
```
注: wandb中 `best_test_score` 字段仅用于内部跟踪，不作为报告指标。

**汇总** (仅以 Top1 为准):
- Top1 Val: **60.00%** (Iter 1)
- **Top1 Test: 43.33%**
- Top5 Avg Val: 56.00%
- Top5 Avg Test: 43.47%
- Val-Test Gap (Top1): **+16.67%**

### 对比总表 (仅报告 Top1)

| 指标 | Baseline | v1 | v2 | v2 vs v1 |
|------|----------|-----|-----|----------|
| Top1 Val | 42.22% | 73.33% | 60.00% | -13.33% |
| **Top1 Test** | **49.33%** | 46.00% | 43.33% | **-2.67%** |
| Top5 Avg Val | - | 58.67% | 56.00% | -2.67% |
| Top5 Avg Test | - | 42.67% | 43.47% | +0.80% |
| Val-Test Gap (Top1) | -7.11% | +27.33% | +16.67% | **-10.66%** (改善) |

⚠️ **指标选择说明**: 本报告不使用 "Best Test" (在test集上cherry-pick最高分的prompt) 作为主要指标，因为这会造成 test set leak。唯一有意义的指标是 **Top1 Test** — 由验证集选出的最佳prompt在test上的表现。

---

## Wandb元数据

### v1 wandb.init config

```python
{
    "num_iterations": 20,
    "num_fewshot_examples": 3,
    "top_k_prompts": 5,
    "val_sample_size": 15,
    "num_threads": 8,
    "solver_model": "openrouter/openai/gpt-4.1-mini",
    "generator_model": "openrouter/openai/gpt-4.1-mini",
    "base_prompt": "(none)",
    "framework": "litellm (no DSPy, no base prompt)",
    "trainset_size": 45,
    "valset_size": 45,
    "testset_size": 150,
}
```
- Project: `aime-math-litellm-agnostic-nb`
- Run name: `litellm_gpt41mini_20iter_3shot`

### v2 wandb.init config

```python
{
    "version": "v2-fixed-val",
    "num_iterations": 20,
    "num_fewshot_examples": 3,
    "top_k_prompts": 5,
    "val_size": 15,
    "val_strategy": "fixed_head_slice_of_paper_valset",
    "val_sample_size": 15,
    "split_alignment": "identical_to_paper_v1",
    "num_threads": 8,
    "solver_model": "openrouter/openai/gpt-4.1-mini",
    "generator_model": "openrouter/openai/gpt-4.1-mini",
    "base_prompt": "(none)",
    "framework": "litellm (no DSPy, no base prompt, fixed val)",
    "trainset_size": 45,
    "full_valset_size": 45,
    "valset_size": 15,
    "testset_size": 150,
}
```
- Project: `aime-math-litellm-agnostic-nb`
- Run name: `litellm_v2_gpt41mini_20iter_3shot`

---

## 优缺点分析

### v1 (每轮随机采样)

**优点**:
- 每个prompt在不同的验证子集上评估，降低单一子集带来的偏差
- 理论上更接近真实分布的蒙特卡洛估计
- 与原始论文/dynamic_fewshot.py保持一致

**缺点**:
- **选择不公平**: Rank 1 和 Rank 2 可能评估在完全不同的问题上
- **val_score不可比**: 某个prompt的"高分"可能是因为抽到了更简单的问题
- 验证方差大 (n=15 样本独立采样)
- Val-Test Gap (Top1) 严重放大 (+27.33%)

### v2 (固定头切片)

**优点**:
- ✅ **公平比较**: 所有候选prompt在完全相同的15个问题上评估
- ✅ **低选择噪声**: 排名完全基于prompt质量，而非抽样偏差
- ✅ **Val-Test Gap缩小**: Top1 gap 从 v1 的 +27.33% 降到 +16.67% (-10.66%)
- ✅ **Top5 Avg Test略高** (+0.80%)，表明选择更靠谱

**缺点**:
- 15题的固定子集可能不能代表全45题分布
- 过度拟合到头切片 (但实践中差距不大)
- 理论上不如v1的随机采样"无偏"

### 结论

**v2 是更严谨的实验设置**，建议作为默认选择：

1. **复现论文**: v1保持了原始论文random.sample行为，如果严格复现请用v1
2. **研究分析**: v2提供了公平的prompt排名，Val-Test Gap更小，适合研究prompt质量
3. **GPT-4.1-mini优化未成功**: 以 Top1 Test 为准：Baseline 49.33% > v1 46.00% > v2 43.33%。**优化后的prompt 反不如默认baseline**。
4. **不推荐使用 Best test selection**: 那会从 test set 选prompt，造成信息泄漏，不是合法评估

### ⚠️ 实验可复现性警告

两个脚本均**未设置全局PRNG种子** (`random.seed(...)`)。这意味着：

- 重跑v1或v2会产生**不同的fewshot抽样、不同的生成prompt、不同的val子集** (v1)
- 报告中的所有具体分数仅代表**单次运行**结果
- 不能保证Top1/Best排名在重跑下保持稳定 (尤其是15题的小验证集)
- 同一脚本v1间的差异可能大于v1↔v2的差异

建议未来运行：
```python
import random
random.seed(42)  # 在main()开头加入
```
并多次运行 (n≥3) 取均值±标准差报告。

---

## 附录

### 运行命令

```bash
# v1
$env:OPENROUTER_API_KEY_AIME_MATH_GPT41MINI="<key>"
uv run python -m examples.aime_math.dynamic_fewshot_litellm_gpt41mini

# v2
$env:OPENROUTER_API_KEY_AIME_MATH_V2="<key>"
uv run python -m examples.aime_math.dynamic_fewshot_litellm_v2_gpt41mini

# Baseline
$env:OPENROUTER_API_KEY_AIME_MATH_GPT41MINI="<key>"
uv run python -m examples.aime_math.baseline_test_litellm_gpt41mini
```

### 历史运行时间

| 实验 | 运行时间 | 日志大小 |
|------|----------|----------|
| v1 | 2026-05-02 01:27 → 断行 | logs_fewshot_litellm.txt (~截断) |
| v2 | 2026-05-02 02:38 → 05:39 完成 | logs_v2_gpt41mini.txt (365 KB) |
| Baseline | 2026-05-02 01:27 → 完成 | logs_baseline_litellm.txt |

### 关联文件

- v1 脚本: `examples/aime_math/dynamic_fewshot_litellm_gpt41mini.py`
- v2 脚本: `examples/aime_math/dynamic_fewshot_litellm_v2_gpt41mini.py`
- Baseline: `examples/aime_math/baseline_test_litellm_gpt41mini.py`
- 共享utils: (无，v1/v2都是单文件实现)
- Artifact对齐: `examples/aime_math/artifact_aligned.py` (DSPy版本，作为参考)
