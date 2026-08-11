# GEPA 实验结果完整报告

**生成时间**: 2026-05-04  
**实验范围**: AIME v2, LiveBench-Math, IFBench, HoVer

---

## 目录

1. [总体摘要](#总体摘要)
2. [详细结果表](#详细结果表)
3. [Top1 vs Best 对比](#top1-vs-best-对比)
4. [各Benchmark详细配置](#各benchmark详细配置)
5. [关键发现](#关键发现)
6. [优化后的Prompt示例](#优化后的prompt示例)
7. [建议与结论](#建议与结论)

---

## 总体摘要

| Benchmark | 最佳模型 | Top1测试分 | 关键观察 |
|-----------|----------|------------|----------|
| **AIME v2** | Qwen3-8B | 66.00% | Qwen3-8B显著领先GPT-4.1-mini (+24%) |
| **LiveBench-Math** | Qwen3-8B | 30.16% | 两模型差距较小 (+3.18%) |
| **IFBench** | GPT-4.1-mini | 51.53% | GPT-4.1-mini领先 (+10.71%) |
| **HoVer** | GPT-4.1-mini | 45.67% | GPT-4.1-mini略领先 (+1.67%) |

---

## 详细结果表

### AIME v2 (数学竞赛)

| 实验 | 模型 | 类型 | 迭代次数 | Top1验证分 | **Top1测试分** | Val-Test Gap |
|------|------|------|----------|------------|----------------|--------------|
| AIME v2 | GPT-4.1-mini | 动态Few-shot | 12 | 53.33% (Iter 2) | **42.00%** | 11.33% |
| AIME v2 | Qwen3-8B | 动态Few-shot | 20 | 73.33% (Iter 18) | **66.00%** | 7.33% |

**数据集配置**:
- Dataset: AI-MO/aimo-validation-aime + MathArena/aime_2025
- Split: 45 train / 45 val / 150 test
- Validation: 固定前15个样本 (head-slice)
- Metric: 最终答案精确匹配

**模型配置**:
- GPT-4.1-mini: `temperature=1.0, top_p=0.95, top_k=20, max_tokens=16384`
- Qwen3-8B: `temperature=0.6, top_p=0.95, top_k=20, provider=alibaba`

---

### LiveBench-Math

| 实验 | 模型 | 类型 | Top1验证分 | **Top1测试分** | Val-Test Gap |
|------|------|------|------------|----------------|--------------|
| LiveBench | GPT-4.1-mini | 动态Few-shot | 33.33% (Iter 1) | **26.98%** | 6.35% |
| LiveBench | Qwen3-8B | 动态Few-shot | 36.67% (Iter 3) | **30.16%** | 6.51% |

**数据集配置**:
- Dataset: livebench/math (368 total)
- Split: 121 train / 121 val / 126 test (shuffled seed 0)
- Tasks: AMPS_Hard, math_comp, olympiad
- Validation: 固定前30个样本 (head-slice from full val)

**模型配置**:
- GPT-4.1-mini: `temperature=1.0`
- Qwen3-8B: `temperature=0.6, top_p=0.95, top_k=20, provider=alibaba`

---

### IFBench (指令遵循)

| 实验 | 模型 | 类型 | 迭代次数 | Top1验证分 | **Top1测试分** | Val-Test Gap |
|------|------|------|----------|------------|----------------|--------------|
| IFBench | GPT-4.1-mini | 单阶段优化 | 12 | 96.67% (Iter 12) | **51.53%** | 45.14% |
| IFBench | GPT-4.1-mini | 两阶段优化 | 4 | 93.33% (Iter 4) | **48.81%** | 44.52% |
| IFBench | GPT-4.1-mini | Base (DSPy) | - | - | **48.30%** | - |
| IFBench | Qwen3-8B | 单阶段优化 | 2 | 90.00% (Iter 2) | **40.82%** | 49.18% |
| IFBench | Qwen3-8B | 两阶段优化 | 8 | 88.33% (Iter 8) | **43.88%** | 44.45% |
| IFBench | Qwen3-8B | Retry | 10 | 88.33% (Iter 10) | **38.95%** | 49.38% |
| IFBench | Qwen3-8B | Base (DSPy) | - | - | **38.61%-41.50%** | - |

**数据集配置**:
- Dataset: gepa-artifact IFBench (`IFBench_train.jsonl`, `IFBench_test.jsonl`)
- Schema: `prompt` input, `response` output
- Split: 150 train (from 300 val pool, seed 1) / 300 val / 294 test
- Program: 2-stage predictors (drafter + finalizer)
- Metric: 任务特定指标 (accuracy, edit distance, etc.)

---

### HoVer (多跳检索)

| 实验 | 模型 | 类型 | 迭代次数 | Top1验证分 | **Top1测试分** | Val-Test Gap |
|------|------|------|----------|------------|----------------|--------------|
| HoVer | GPT-4.1-mini | V2 | 2 | 60.00% (Iter 2) | **45.67%** | 14.33% |
| HoVer | Qwen3-8B | V2 | 8 | 53.33% (Iter 8) | **44.00%** | 9.33% |
| HoVer | GPT-4.1-mini | Base (DSPy) | - | - | **36.33%** | - |

**数据集配置**:
- Dataset: HuggingFace 'hover' (3-hop only, `count_unique_docs == 3`)
- Split: 150 train / 300 val / test (shuffled seed 0/1)
- Validation: 固定前30个样本 (head-slice from 300 val)
- Program: 4-predictor multi-hop
  - `summarize1`: 总结第一轮检索结果
  - `create_query_hop2`: 生成第二轮查询
  - `summarize2`: 总结第二轮结果
  - `create_query_hop3`: 生成第三轮查询
- Metric: `discrete_retrieval_eval` (gold_titles ⊆ found_titles after normalize_text)

---

## Top1 vs Best 对比

| 实验 | 模型 | Top1测试分 | Best测试分 | 差距 | 备注 |
|------|------|------------|------------|------|------|
| AIME v2 | GPT-4.1-mini | 42.00% | 49.33% | -7.33% | Top5平均选择更有效 |
| AIME v2 | Qwen3-8B | 66.00% | 67.33% | -1.33% | 相对稳定 |
| LiveBench | GPT-4.1-mini | 26.98% | 32.54% | -5.56% | 存在波动 |
| LiveBench | Qwen3-8B | 30.16% | 33.33% | -3.17% | 相对稳定 |
| IFBench (单阶段) | GPT-4.1-mini | 51.53% | 51.53% | 0% | 使用Top1选择 |
| IFBench (两阶段) | GPT-4.1-mini | 48.81% | 48.81% | 0% | 使用Top1选择 |
| IFBench (单阶段) | Qwen3-8B | 40.82% | 40.82% | 0% | 使用Top1选择 |
| IFBench (两阶段) | Qwen3-8B | 43.88% | 43.88% | 0% | 使用Top1选择 |
| HoVer | GPT-4.1-mini | 45.67% | 45.67% | 0% | 使用Top1选择 |
| HoVer | Qwen3-8B | 44.00% | 44.00% | 0% | 使用Top1选择 |

**观察**: AIME和LiveBench使用Top5平均选择，Top1与Best存在差距；IFBench和HoVer使用Top1选择，两者一致。

---

## 各Benchmark详细配置

### AIME v2

```yaml
Dataset: AI-MO/aimo-validation-aime
Split:
  train: 45
  val: 45
  test: 150 (MathArena/aime_2025)
Validation: Fixed head-slice of 15 from 45 val
Metric: Exact match on final answer
Shuffling: random.Random(0)
```

### LiveBench-Math

```yaml
Dataset: livebench/math
Total: 368
Split:
  train: 121
  val: 121
  test: 126
Tasks:
  - AMPS_Hard: LaTeX数学表达式
  - math_comp: 3位整数或5次重复选择
  - olympiad: 逗号分隔的表达式索引
Validation: Fixed head-slice of 30 from 121 val
Shuffling: random.Random(0)
```

### IFBench

```yaml
Dataset: gepa-artifact IFBench
Files:
  - IFBench_train.jsonl
  - IFBench_test.jsonl
Schema:
  input: prompt
  output: response
Split:
  train: 150 (from 300 val pool, random.Random(1).sample)
  val: 300 (head slice)
  test: 294 (capped at 300)
Program: 2-stage (drafter + finalizer)
Metric: Task-specific (accuracy, edit distance, constraints satisfaction)
```

### HoVer

```yaml
Dataset: HuggingFace 'hover'
Filter: count_unique_docs == 3 (3-hop only)
Split:
  train: 150
  val: 300
  test: remaining
Shuffling: random.Random(0) for split, random.Random(1).sample for trim
Validation: Fixed head-slice of 30 from 300 val
Program: 4-predictor multi-hop
  - summarize1
  - create_query_hop2
  - summarize2
  - create_query_hop3
Metric: discrete_retrieval_eval
  - gold_titles ⊆ found_titles after normalize_text
Retriever: BM25 (wiki.abstracts.2017.tar.gz ~5GB)
```

---

## 关键发现

### 1. 模型性能对比

| Benchmark | GPT-4.1-mini | Qwen3-8B | 领先方 | 差距 |
|-----------|--------------|----------|--------|------|
| AIME v2 | 42.00% | 66.00% | Qwen3-8B | +24.00% |
| LiveBench | 26.98% | 30.16% | Qwen3-8B | +3.18% |
| IFBench | 51.53% | 40.82% | GPT-4.1-mini | +10.71% |
| HoVer | 45.67% | 44.00% | GPT-4.1-mini | +1.67% |

**结论**:
- **数学任务(AIME, LiveBench)**: Qwen3-8B表现更优
- **指令遵循(IFBench)**: GPT-4.1-mini表现更优
- **多跳检索(HoVer)**: 两模型接近，GPT-4.1-mini略优

### 2. 过拟合分析

| Benchmark | 典型Val-Test Gap | 严重程度 | 原因分析 |
|-----------|------------------|----------|----------|
| IFBench | 44-49% | **严重** | 验证集规模小(300)，任务多样性强 |
| HoVer | 9-14% | 中等 | 验证集规模适中(300) |
| AIME | 7-11% | 可控 | 验证集小(15)，但任务单一 |
| LiveBench | 6% | **健康** | 验证集规模合适(121)，与测试分布一致 |

### 3. 优化策略效果

**IFBench - GPT-4.1-mini**:
- 单阶段优化: 51.53% (vs Base 48.30%, +3.23%)
- 两阶段优化: 48.81% (vs Base 48.30%, +0.51%)
- **结论**: 单阶段优化更有效

**IFBench - Qwen3-8B**:
- 单阶段优化: 40.82%
- 两阶段优化: 43.88%
- Retry: 38.95%
- Base: 38.61%-41.50%
- **结论**: 两阶段优化略优

**HoVer**:
- GPT-4.1-mini V2: 45.67% (vs Base 36.33%, +9.34%)
- Qwen3-8B V2: 44.00% (vs Base ~36%, +8%)
- **结论**: 4-predictor优化显著提升

---

## 优化后的Prompt示例

### HoVer (Qwen3-8B) - 4阶段优化

#### summarize1
```
Summarize the retrieved passages in 3-5 concise sentences, highlighting 
key entities, their relationships, and any unresolved references that the 
next query must address. Focus on extracting named entities, core concepts, 
and contextual details directly tied to the claim.
```

#### create_query_hop2
```
Write a BM25 query that connects the claim to the first summary's entities 
using disambiguated terms and relational phrasing. Prioritize entity bridges 
(e.g., "X is associated with Y") and contextual modifiers (e.g., "historical 
context of Z") to retrieve documents that expand on unresolved references from 
the first summary.
```

#### summarize2
```
Summarize the second-hop passages, integrating the claim and first summary's 
context. Extract new entities, relations, and unresolved references that the 
final query must resolve. Emphasize contradictions, supporting evidence, and 
cross-references between documents to guide the final retrieval step.
```

#### create_query_hop3
```
Craft a BM25 query that synthesizes the claim, first summary, and second 
summary to target remaining gold documents. Use precise entity combinations, 
temporal markers, and disambiguated terms to ensure full coverage of all 
supporting evidence, prioritizing documents that resolve unresolved references 
from prior stages.
```

### IFBench - 两阶段 (GPT-4.1-mini)

#### Stage1 (Drafter)
```
You are the Drafter. Your task is to read the user's query carefully and 
produce an initial answer that follows all the instructions embedded in 
that query. Before writing your response, analyze and reason about all the 
constraints, formats, styles, and specific content requirements given. Ensure 
your response adheres strictly to instructions such as length constraints, 
formatting rules, case transformations, repetition or structural demands, and 
any requested content details.
```

#### Stage2 (Finalizer)
```
You are the Finalizer. Your input includes the user's query and the Drafter's 
initial response. Your task is to carefully review both, identifying any 
inaccuracies, omissions, or deviations from the instructions in the Drafter's 
response. Thoroughly reason about each specified constraint, formatting 
requirement, and content instruction given by the user. Then produce a final 
response that fully complies with all instructions.
```

### LiveBench (Qwen3-8B) - 最佳单阶段

```
Always think step-by-step, justify your reasoning, and place the final 
answer inside \boxed{}. If uncertain, provide your best guess but ensure 
strict adherence to the required format. Refer to the examples provided 
for formatting patterns.
```

---

## 建议与结论

### 模型选择建议

| 任务类型 | 推荐模型 | 原因 |
|----------|----------|------|
| 数学竞赛(AIME) | **Qwen3-8B** | +24%性能优势 |
| 综合数学(LiveBench) | **Qwen3-8B** | +3%性能优势，相对稳定 |
| 指令遵循(IFBench) | **GPT-4.1-mini** | +10%性能优势 |
| 多跳检索(HoVer) | **GPT-4.1-mini** | 略优，两种都可用 |

### 优化策略建议

1. **解决IFBench过拟合**:
   - 减少验证集规模或增加正则化
   - 考虑使用早停(early stopping)
   - 增加验证集多样性

2. **AIME优化**:
   - 优先使用Qwen3-8B
   - Top5选择策略比Top1更有效
   - 考虑增加验证集规模(目前仅15)

3. **LiveBench优化**:
   - 验证-测试gap健康(6%)，优化策略有效
   - 可考虑增加迭代次数

4. **HoVer优化**:
   - 4-predictor优化策略效果显著(+9%)
   - 各阶段prompt独立优化有价值

### 未来工作

1. 探索**模型融合**策略(Qwen3-8B for math, GPT-4.1-mini for IF)
2. 研究**验证集选择策略**对过拟合的影响
3. 尝试**多模型集成**提升稳定性
4. 优化**两阶段程序**的设计，缩小单阶段差距

---

## 附录: 原始Log文件列表

```
# AIME
logs_v2_gpt41mini.txt
logs_v2_qwen.txt

# LiveBench
logs_lb_gpt41mini.txt
logs_lb_qwen.txt

# IFBench
logs_if_artifact_gpt41mini_retry.txt
logs_if_artifact_gpt41mini_twostage.txt
logs_if_artifact_qwen.txt
logs_if_artifact_qwen_retry.txt
logs_if_artifact_qwen_twostage.txt
logs_if_base_gpt41mini.txt
logs_if_base_qwen.txt

# HoVer
logs_hover_v2_gpt41mini.txt
logs_hover_v2_qwen.txt
logs_hover_base_gpt41mini.txt
```
