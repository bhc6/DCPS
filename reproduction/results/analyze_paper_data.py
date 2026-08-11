import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------
# 0. 环境与路径设置
# ---------------------------------------------------------
RESULTS_DIR = os.environ.get(
    "DCPS_RESULTS_DIR", os.path.dirname(os.path.abspath(__file__))
)
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")
RAW_DATA_PATH = os.path.join(RESULTS_DIR, "raw_wandb_data.csv")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# 设置 Seaborn 论文绘图风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# ---------------------------------------------------------
# 0.5 加载与清洗 WandB 真实数据
# ---------------------------------------------------------
DATASET_KEYWORDS = [
    ('AIME', 'AIME-2025'),
    ('LIVEBENCH', 'LIVEBENCH-MATH'),
    ('HOTPOT', 'HOTPOTQA'),
    ('IFBENCH', 'IFBENCH'),
    ('HOVER', 'HOVER'),
    ('PUPA', 'PUPA'),
    ('PAPILLON', 'PUPA'),
]

def _match_dataset(text):
    if not text:
        return None
    s = str(text).upper()
    for kw, ds in DATASET_KEYWORDS:
        if kw in s:
            return ds
    return None

def load_and_clean_data():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"[Warning] 找不到原始数据文件 {RAW_DATA_PATH}，请先运行 fetch_wandb_data.py")
        return None
        
    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_clean = pd.DataFrame()
    
    # 提取基本元数据
    df_clean['run_id'] = df_raw['run_id']
    df_clean['project'] = df_raw['project']
    df_clean['run_name'] = df_raw.get('run_name', '')
    
    # 统一数据集名称：依次从 config_dataset / config_benchmark_name / run_name / project 推断
    def clean_dataset(row):
        for col in ('config_dataset', 'config_benchmark_name'):
            if col in row and pd.notna(row[col]):
                m = _match_dataset(row[col])
                if m: return m
                return str(row[col]).upper()
        m = _match_dataset(row.get('run_name'))
        if m: return m
        m = _match_dataset(row.get('project'))
        if m: return m
        return 'UNKNOWN'
        
    df_clean['dataset'] = df_raw.apply(clean_dataset, axis=1)
    df_clean['seed'] = pd.to_numeric(df_raw.get('config_seed', 0), errors='coerce')
    df_clean['runtime_mins'] = pd.to_numeric(df_raw.get('runtime_mins', 0.0), errors='coerce')
    
    # 提取 top_k 用于真实绘制 Best-of-N 图表
    df_clean['top_k'] = pd.to_numeric(df_raw.get('config_top_k', df_raw.get('config_topk', 1)), errors='coerce').fillna(1)
    
    # 统一模型名称：去掉 openrouter/ 前缀，保留 gpt-4o-mini 独立标签
    def determine_model(row):
        candidates = []
        for col in ['config_solver_model', 'config_model', 'config_agent_model',
                    'config_target_model', 'config_lm_name', 'config_generator_model']:
            if col in row and pd.notna(row[col]):
                candidates.append(str(row[col]))
        # 从 run_name 中反推断
        rn = str(row.get('run_name', '')).lower()
        if rn:
            candidates.append(rn)
        for raw in candidates:
            v = raw.lower().replace('openrouter/', '').replace('openai/', '')
            if 'qwen3-8b' in v or 'qwen/qwen3-8b' in v: return 'Qwen3-8B'
            if 'gpt-4.1-mini' in v or 'gpt-41-mini' in v: return 'GPT-4.1-mini'
            if 'gpt-4o-mini' in v: return 'GPT-4o-mini'
            if 'gemma-1.1-7b' in v: return 'google/gemma-1.1-7b-it'
            if 'gemma-1.1-2b' in v: return 'google/gemma-1.1-2b-it'
        # 返回原始首选（便于后续诊断）
        if candidates:
            return candidates[0].replace('openrouter/', '')
        return 'Unknown'
        
    df_clean['agent_model'] = df_raw.apply(determine_model, axis=1)
    
    # 用于检测某 run 是否真实运行了 PPO（任意 PPO 计数器/计时字段存在即视为真）
    _PPO_SIGNAL_COLS = [
        'summary_global_step', 'summary_step',
        'summary_ppo/time/ppo/total', 'summary_PPO/time/ppo/total',
        'summary_time/ppo/total', 'summary_ppo/time/ppo/optimize_step',
        'summary_PPO/time/ppo/optimize_step', 'summary_time/ppo/optimize_step',
        'summary_final/total_model_updates',
    ]
    _PPO_SIGNAL_COLS = [c for c in _PPO_SIGNAL_COLS if c in df_raw.columns]

    def _ppo_ran(row):
        for c in _PPO_SIGNAL_COLS:
            if pd.notna(row.get(c)):
                return True
        return False

    # 构建方法标识 (Method Mapping)
    # 关键修正：StablePrompt 代码库可在启用 PPO 与禁用 PPO 两种模式下运行；
    #          后者在论文/用户口径中常被称为 GFB / AlgPrompt（同代码库去 PPO）。
    #          以 PPO 计数器/计时字段是否存在作为硬区分。
    def determine_method(row):
        project = str(row.get('project', '')).lower()
        run_name_l = str(row.get('run_name', '')).lower()

        if 'dynamic-fewshot' in project:
            return 'FSPO'
        if project == 'gepa':
            opt = row.get('config_optimizer_name')
            if pd.notna(opt):
                if 'Baseline' in str(opt):
                    return 'Baseline'
                return str(opt)
            return 'GEPA'
        if 'baseline' in project:
            return 'Baseline'
        if any(b in run_name_l for b in ['baseline', 'cot', 'zeroshot']):
            return 'Baseline'

        # StablePrompt 代码库特征：同时有 prompt_per_example 与 batch_size
        ppe_present = pd.notna(row.get('config_prompt_per_example'))
        bs_present = pd.notna(row.get('config_batch_size'))
        if ppe_present and bs_present:
            if _ppo_ran(row):
                # PPO 真正运行 → StablePrompt-PPO（cs/ca 为子型）
                cs = row.get('config_cs')
                if pd.notna(cs):
                    try:
                        return f"StablePrompt-PPO(cs={float(cs):g})"
                    except Exception:
                        return "StablePrompt-PPO"
                return "StablePrompt-PPO"
            # 同代码库但 PPO 未运行 → AlgPrompt（=GFB-NoPPO）
            return "AlgPrompt"

        return "Unknown"

    df_clean['method'] = df_raw.apply(determine_method, axis=1)
    
    # 获取分数 (统一为百分制)。
    # 原则：存在但为 0 -> NaN（记为失败 run，不参与均值）；
    #       0 < val <= 1.0 视为比例，乘 100；val > 1.0 视为百分。
    score_cols = [
        # StablePrompt-PPO / GFB
        'summary_final/best_accuracy',
        'summary_best_accuracy',
        'summary_final/mean_accuracy',
        'summary_mean_accuracy',
        # FSPO / GEPA / dynamic-fewshot
        'summary_best_test_score',
        'summary_test_score',
        'summary_test/score',
        'summary_test/avg_test_score',
        # AlgPrompt（同代码库 PPO 禁用）
        'summary_final_acc',
        'summary_final_mean_acc',
        'summary_test_acc',
        'summary_valid_acc',
        # 通用
        'summary_metrics/accuracy',
        'summary_accuracy',
    ]
    
    def get_score(row):
        for col in score_cols:
            if col in row and pd.notna(row[col]):
                val = float(row[col])
                if val == 0:
                    return np.nan  # 失败 run，不参与后续统计
                if 0 < val <= 1.0:
                    return val * 100.0
                return val
        return np.nan
        
    df_clean['final_score'] = df_raw.apply(get_score, axis=1)
    
    # 计算真实 Rollouts。按论文定义：1 rollout = 系统对一个样本完整执行一次并评分。
    # 严格原则：只从真实记录及其确定性推导；预算上限不计为实际 rollouts。
    def _num(row, *keys):
        for k in keys:
            if k in row and pd.notna(row[k]):
                try:
                    v = float(row[k])
                    if v > 0:
                        return v
                except Exception:
                    pass
        return np.nan

    # PPO 是否实际运行的错额判定：任意 PPO 计数器/计时字段存在即为真
    PPO_SIGNAL_COLS = [
        'summary_global_step', 'summary_step',
        'summary_ppo/time/ppo/total', 'summary_PPO/time/ppo/total',
        'summary_time/ppo/total', 'summary_ppo/time/ppo/optimize_step',
        'summary_PPO/time/ppo/optimize_step', 'summary_time/ppo/optimize_step',
        'summary_final/total_model_updates',
    ]
    existing_ppo_cols = [c for c in PPO_SIGNAL_COLS if c in df_raw.columns]

    def _ppo_actually_ran(row):
        for c in existing_ppo_cols:
            v = row.get(c)
            if pd.notna(v):
                return True
        return False

    df_clean['ppo_used'] = df_raw.apply(_ppo_actually_ran, axis=1)

    def _is_stableprompt_codebase(row):
        # StablePrompt 代码库特征：同时存在 prompt_per_example 与 batch_size 配置项
        # （可能启用 PPO也可能禁用 PPO，后者在论文/用户习惯中常被叫作 GFB / AlgPrompt）
        return (pd.notna(row.get('config_prompt_per_example')) and
                pd.notna(row.get('config_batch_size')))

    def calculate_rollouts(row):
        # 1) wandb 直接记录的总样本评估次数（最优先，不仅限于 PPO）
        v = _num(row, 'summary_total_metric_calls', 'summary_total_lm_calls', 'summary_rollouts')
        if pd.notna(v):
            return v

        project = str(row.get('project', '')).lower()

        # 2) FSPO（dynamic-fewshot 项目）
        if 'dynamic-fewshot' in project:
            iters = _num(row, 'summary_total_iterations', 'summary_iteration',
                         'config_num_iterations', 'config_iterations')
            kp = _num(row, 'config_top_k_prompts', 'config_top_k')
            vs = _num(row, 'config_valset_size', 'config_val_size', 'config_full_valset_size')
            if pd.notna(iters) and pd.notna(kp) and pd.notna(vs):
                return iters * kp * vs
            return np.nan

        # 3) StablePrompt 代码库（PPO 启用）：用实际完成的 PPO 步数
        #    rollouts = (global_step + 1) * batch_size * prompt_per_example
        if _is_stableprompt_codebase(row) and _ppo_actually_ran(row):
            ppe = _num(row, 'config_prompt_per_example')
            bs = _num(row, 'config_batch_size')
            gstep = _num(row, 'summary_global_step', 'summary_step')
            if pd.notna(gstep) and pd.notna(bs) and pd.notna(ppe):
                return (gstep + 1.0) * bs * ppe
            # 某些 PPO run 有计时但未录 step：回退调度上限 epochs * train * ppe
            epochs = _num(row, 'config_epochs')
            train = _num(row, 'config_train_data_size', 'config_train_size')
            if pd.notna(epochs) and pd.notna(train) and pd.notna(ppe):
                return epochs * train * ppe
            return np.nan

        # 4) StablePrompt 代码库 (PPO 禁用 = AlgPrompt / GFB 变体)。
        #    此类 run 仅在 sample_total_metric_calls 有值时才能得到真实 rollouts，上面步 1 已处理。
        #    部分项目不记 train_data_size，无法从 config 推导 → 返回 NaN，不估算。
        return np.nan

    df_clean['rollouts'] = df_raw.apply(calculate_rollouts, axis=1)
    df_clean['rollouts_budget'] = df_raw.apply(lambda r: _num(r, 'config_max_metric_calls'), axis=1)

    # 例外保留原有预算作为一个 *参考* 上界，供下游仅在完全缺失时参考。
    # 提取其他关键实验区分字段
    for src, dst in [
        ('config_cs', 'cs'),
        ('config_ca', 'ca'),
        ('config_metric', 'metric'),
        ('config_optimizer_name', 'optimizer_name'),
        ('config_optimization_surface', 'optimization_surface'),
        ('config_metaprompt_style', 'metaprompt_style'),
        ('config_num_fewshot_examples', 'num_fewshot'),
        ('config_batch_size', 'batch_size'),
        ('config_prompt_per_example', 'prompt_per_example'),
        ('config_train_data_size', 'train_size'),
        ('config_epochs', 'epochs'),
        ('config_top_k_prompts', 'top_k_prompts'),
        ('config_valset_size', 'valset_size'),
        ('config_num_iterations', 'num_iterations'),
        ('summary_global_step', 'global_step'),
        ('summary_total_iterations', 'total_iterations'),
    ]:
        if src in df_raw.columns:
            df_clean[dst] = df_raw[src]
    
    # 测试集大小（供下游 RCEI 的 t_b 使用）
    def get_test_size(row):
        for col in ('config_test_size', 'config_testset_size', 'config_test_data_size',
                    'config_final_eval_size', 'config_num_test_example', 'summary_test_data_size'):
            if col in row and pd.notna(row[col]):
                try:
                    v = float(row[col])
                    if v > 0:
                        return v
                except Exception:
                    pass
        return np.nan
    df_clean['test_size'] = df_raw.apply(get_test_size, axis=1)
        
    # 提取输入输出 Token 数量
    def get_input_tokens(row):
        # 针对 GEPA/MIPROv2 将 optimizer 和 test 分开记录的情况
        opt_in = float(row.get('summary_optimizer/input_tokens', 0) if pd.notna(row.get('summary_optimizer/input_tokens')) else 0)
        test_in = float(row.get('summary_test/input_tokens', 0) if pd.notna(row.get('summary_test/input_tokens')) else 0)
        if opt_in > 0 or test_in > 0:
            return opt_in + test_in
        
        # 针对 FSPO 或其他使用常规 cumulative 记录的情况
        for col in ['summary_input_tokens_cumulative', 'summary_usage/prompt_tokens', 'summary_total_prompt_tokens']:
            if col in row and pd.notna(row[col]):
                return float(row[col])
                
        # 针对本地模型的粗略估算
        if 'summary_epoch_summary/mean_prompt_length' in row and pd.notna(row['summary_epoch_summary/mean_prompt_length']):
            rollouts = df_clean.loc[row.name, 'rollouts']
            return float(row['summary_epoch_summary/mean_prompt_length']) * float(rollouts if pd.notna(rollouts) else 0)
            
        return np.nan

    def get_output_tokens(row):
        opt_out = float(row.get('summary_optimizer/output_tokens', 0) if pd.notna(row.get('summary_optimizer/output_tokens')) else 0)
        test_out = float(row.get('summary_test/output_tokens', 0) if pd.notna(row.get('summary_test/output_tokens')) else 0)
        if opt_out > 0 or test_out > 0:
            return opt_out + test_out
                
        for col in ['summary_output_tokens_cumulative', 'summary_usage/completion_tokens', 'summary_total_completion_tokens']:
            if col in row and pd.notna(row[col]):
                return float(row[col])
        return np.nan

    df_clean['input_tokens'] = df_raw.apply(get_input_tokens, axis=1)
    df_clean['output_tokens'] = df_raw.apply(get_output_tokens, axis=1)

    # API Cost：
    # 1) 优先使用 wandb 中记录的真实美元 (summary_optimizer/cost + summary_test/cost)。
    # 2) 其次对 GPT 系 API 模型按 token 价格表估算。
    # 3) 本地模型 (Qwen / Gemma) 统一记为 NaN。
    PRICE_TABLE = {  # USD per 1M tokens (input, output)
        'GPT-4.1-mini': (0.15, 0.60),
        'GPT-4o-mini': (0.15, 0.60),
    }
    def get_real_cost(row):
        opt_c = pd.to_numeric(row.get('summary_optimizer/cost'), errors='coerce')
        test_c = pd.to_numeric(row.get('summary_test/cost'), errors='coerce')
        total = 0.0
        seen = False
        if pd.notna(opt_c):
            total += float(opt_c); seen = True
        if pd.notna(test_c):
            total += float(test_c); seen = True
        return total if seen else np.nan
    df_clean['real_cost_usd'] = df_raw.apply(get_real_cost, axis=1)

    def calculate_cost(row):
        if pd.notna(row.get('real_cost_usd')):
            return row['real_cost_usd']
        model = row.get('agent_model')
        price = PRICE_TABLE.get(model)
        if price is None:
            return np.nan  # 本地模型或未知模型 -> 不估算
        if pd.isna(row['input_tokens']) or pd.isna(row['output_tokens']):
            return np.nan
        in_p, out_p = price
        return (row['input_tokens'] / 1e6) * in_p + (row['output_tokens'] / 1e6) * out_p

    df_clean['estimated_cost_usd'] = df_clean.apply(calculate_cost, axis=1)

    # 保存清洗后的数据供后续使用
    clean_path = os.path.join(RESULTS_DIR, "clean_paper_data.csv")
    df_clean.to_csv(clean_path, index=False)
    print(f"Successfully cleaned data: {len(df_clean)} runs processed.")
    return df_clean

# ---------------------------------------------------------
# 1. 统计协议实现
# ---------------------------------------------------------
def calculate_paired_statistics(rl_scores, rae_scores, n_resamples=1000):
    rl_scores = np.array(rl_scores)
    rae_scores = np.array(rae_scores)
    diffs = rl_scores - rae_scores
    
    mean_diff = np.mean(diffs)
    
    resampled_diffs = np.random.choice(diffs, size=(n_resamples, len(diffs)), replace=True)
    resampled_means = np.mean(resampled_diffs, axis=1)
    ci_lower = float(np.percentile(resampled_means, 2.5))
    ci_upper = float(np.percentile(resampled_means, 97.5))
    
    try:
        _, p_val = stats.ttest_rel(rl_scores, rae_scores)
    except:
        p_val = 1.0
        
    cohens_d = mean_diff / np.std(diffs, ddof=1) if len(diffs) > 1 and np.std(diffs, ddof=1) != 0 else 0
    
    return {
        "mean_diff": mean_diff,
        "ci_95": (ci_lower, ci_upper),
        "cohens_d": cohens_d,
        "p_value": p_val
    }

# ---------------------------------------------------------
# 2. 绘制 Figures
# ---------------------------------------------------------
def plot_best_of_n_scaling(df_clean):
    if df_clean is None or df_clean.empty:
        return

    # 限定 (FSPO, LIVEBENCH-MATH)；选取样本最多的 agent_model 以避免跨模型混淆
    fspo_all = df_clean[(df_clean['method'] == 'FSPO') & (df_clean['dataset'] == 'LIVEBENCH-MATH')]
    if fspo_all.empty:
        print("[Warning] No FSPO data on LIVEBENCH-MATH for Best-of-N plot.")
        return
    main_model = fspo_all['agent_model'].value_counts().idxmax()
    fspo_data = fspo_all[fspo_all['agent_model'] == main_model]
    stats = fspo_data.groupby('top_k')['final_score'].agg(['mean', 'std', 'count']).dropna(subset=['mean']).sort_index()

    if stats.empty or len(stats) < 2:
        print(f"[Warning] FSPO Best-of-N has insufficient distinct top_k values on LIVEBENCH-MATH ({main_model}).")
        return

    n_values = stats.index.tolist()
    fspo_scores_mean = stats['mean'].tolist()
    fspo_scores_std = stats['std'].fillna(0).tolist()

    gepa_data = df_clean[(df_clean['method'] == 'GEPA') &
                         (df_clean['dataset'] == 'LIVEBENCH-MATH') &
                         (df_clean['agent_model'] == main_model)]
    gepa_score = gepa_data['final_score'].mean() if not gepa_data.empty else np.nan
    gepa_rollouts = gepa_data['rollouts'].mean() if not gepa_data.empty else np.nan

    plt.figure(figsize=(8, 5))
    plt.errorbar(n_values, fspo_scores_mean, yerr=fspo_scores_std, fmt='-o', 
                 capsize=5, label='FSPO (Ours)', color='#1f77b4', linewidth=2)
    
    if pd.notna(gepa_score):
        gepa_label = f'GEPA ({gepa_rollouts:.0f} rollouts)' if pd.notna(gepa_rollouts) else 'GEPA'
        plt.axhline(y=gepa_score, color='#ff7f0e', linestyle='--', label=gepa_label)
    plt.suptitle(f'Model: {main_model}', y=0.99, fontsize=9, color='gray')
    
    plt.xscale('log')
    plt.xticks(n_values, labels=[str(n) for n in n_values])
    plt.xlabel('Number of Sampled Candidates (N)', fontweight='bold')
    plt.ylabel('Best Test Score (LiveBench-Math)', fontweight='bold')
    plt.title('Best-of-N Scaling: Search Compute vs Performance', pad=15)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    save_path = os.path.join(FIG_DIR, 'fig2_best_of_n_scaling.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"[Plot Generated] Best-of-N Scaling -> {save_path}")

def plot_pareto_frontier(df_clean):
    if df_clean is None or df_clean.empty:
        return

    # 选取 PUPA 上能形成 Pareto 边界（至少 2 个方法都有 rollouts+score）的 agent_model
    pupa_all = df_clean[df_clean['dataset'] == 'PUPA']
    if pupa_all.empty:
        print("[Warning] No PUPA data for Pareto Frontier plot.")
        return
    best_model, best_data = None, None
    for model, sub in pupa_all.groupby('agent_model'):
        agg = sub.groupby('method').agg({'rollouts': 'mean', 'final_score': 'mean'}).dropna()
        if len(agg) >= 2 and (best_data is None or len(agg) > len(best_data)):
            best_model, best_data = model, agg
    if best_data is None:
        print("[Warning] No agent_model on PUPA has >=2 methods with both rollouts and score.")
        return
    main_model = best_model
    pupa_data = best_data

    methods = pupa_data.index.tolist()
    rollouts = pupa_data['rollouts'].tolist()
    scores = pupa_data['final_score'].tolist()
    
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(methods))]

    plt.figure(figsize=(8, 5))
    for i in range(len(methods)):
        label_str = methods[i] if methods[i] != 'FSPO' else 'FSPO (Ours)'
        plt.scatter(rollouts[i], scores[i], s=250, color=colors[i], label=label_str, edgecolor='k', zorder=5, alpha=0.9)
        
    sorted_indices = np.argsort(rollouts)
    plt.plot(np.array(rollouts)[sorted_indices], np.array(scores)[sorted_indices], 
             color='grey', linestyle=':', zorder=1, alpha=0.6)
    
    plt.xscale('log')
    plt.xlabel(r'Optimization Cost (Rollouts, log scale) $\downarrow$', fontweight='bold')
    plt.ylabel('Test Score (PUPA)', fontweight='bold')
    plt.title(f'Cost-Performance Pareto Frontier (PUPA, {main_model})', pad=15)
    plt.legend(loc='lower left')
    plt.tight_layout()
    
    save_path = os.path.join(FIG_DIR, 'fig1_pareto_frontier.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"[Plot Generated] Pareto Frontier -> {save_path}")

# ---------------------------------------------------------
# 3. LaTeX 表格生成
# ---------------------------------------------------------
def generate_ablation_latex_table(df_clean):
    if df_clean is None or df_clean.empty:
        return
        
    df_7b = df_clean[df_clean['agent_model'] == 'google/gemma-1.1-7b-it']
    
    # 定义任务族
    glue_tasks = ['MNLI', 'MRPC', 'QNLI', 'RTE', 'SNLI', 'SST2']
    # BBH / II / II_re 中使用的具体子任务名（按实际 run_name 解析后的 dataset 字段）
    bbh_set = {
        'ANTONYMS', 'WORD_IN_CONTEXT', 'RHYMES', 'NUM_TO_VERBAL',
        'ACTIVE_TO_PASSIVE', 'CAUSE_AND_EFFECT', 'FIRST_WORD_LETTER',
        'LARGER_ANIMAL', 'LETTERS_LIST', 'ORTHOGRAPHY_STARTS_WITH',
        'SECOND_WORD_LETTER', 'SENTIMENT', 'SUM', 'SYNONYMS',
        'TAXONOMY_ANIMAL', 'TRANSLATION_EN-DE', 'TRANSLATION_EN-ES',
        'TRANSLATION_EN-FR', 'SINGULAR_TO_PLURAL', 'SENTENCE_SIMILARITY',
        'NEGATION', 'INFORMAL_TO_FORMAL', 'DIFF', 'COMMON_CONCEPT',
        'DYCK_LANGUAGES', 'GENDER_INCLUSIVE_SENTENCES_GERMAN',
        'OBJECT_COUNTING', 'WORD_SORTING', 'OPERATORS', 'TENSE',
        'PRESUPPOSITIONS_AS_NLI', 'LINGUISTICS_PUZZLES', 'HYPERBATON',
        'DISAMBIGUATION_QA', 'EPISTEMIC_REASONING', 'MOVIE_RECOMMENDATION',
        'SNARKS', 'NAVIGATE', 'SPORTS_UNDERSTANDING', 'RUIN_NAMES',
        'IMPLICATURES', 'WINOWHY', 'LOGICAL_FALLACY_DETECTION',
        'BBH-MC', 'BBH-GEN',
    }
    bbh_tasks = [d for d in df_7b['dataset'].unique() if d in bbh_set]
    # MMLU 子任务（覆盖所有学科）
    mmlu_keys = ['HISTORY', 'LAW', 'MMLU', 'BIOLOGY', 'CHEMISTRY', 'PHYSICS',
                 'MATHEMATICS', 'COMPUTER', 'PSYCHOLOGY', 'ECONOMICS',
                 'MEDICINE', 'PHILOSOPHY', 'GEOGRAPHY', 'STATISTICS',
                 'ALGEBRA', 'ANATOMY', 'ASTRONOMY', 'ETHICS', 'KNOWLEDGE',
                 'JURISPRUDENCE', 'MARKETING', 'MANAGEMENT', 'NUTRITION',
                 'POLICY', 'RELATIONS', 'SECURITY', 'SOCIOLOGY', 'VIROLOGY',
                 'GENETICS', 'AGING', 'SEXUALITY', 'PREHISTORY',
                 'ELECTRICAL', 'ECONOMETRICS', 'FACTS', 'LOGIC',
                 'CONCEPTUAL', 'ELEMENTARY', 'CLINICAL', 'PROFESSIONAL',
                 'FORMAL', 'GLOBAL', 'INTERNATIONAL', 'MISCELLANEOUS',
                 'MORAL', 'MACHINE_LEARNING', 'PUBLIC']
    mmlu_tasks = [d for d in df_7b['dataset'].unique()
                  if any(x in str(d) for x in mmlu_keys) and d not in bbh_set]
    
    latex_str = r"""
\begin{table}[ht]
\centering
\caption{Ablation Results across different task families on Gemma1.1-7B.}
\label{tab:gemma_ablation}
\resizebox{\textwidth}{!}{
\begin{tabular}{@{}lccccc@{} }
\toprule
\textbf{Task Family} & \textbf{RL (PPO)} & \textbf{RAE (Ours)} & $\Delta$\textbf{(RL - RAE)} & \textbf{95\% CI} & \textbf{Cohen's d} \\ \midrule
"""
    
    csv_data = []
    MIN_PAIRS = 3  # 低于此样本数不报 CI / Cohen's d
    for family_name, tasks in [('GLUE (Avg)', glue_tasks), ('BBH (Avg)', bbh_tasks), ('MMLU (Avg)', mmlu_tasks)]:
        df_family = df_7b[df_7b['dataset'].isin(tasks)]
        if df_family.empty:
            continue

        # RL = StablePrompt-PPO（任意 cs，PPO 实跑）vs RAE = AlgPrompt（同代码库 PPO 禁用）
        df_family = df_family.copy()
        df_family['method_family'] = df_family['method'].apply(
            lambda m: 'RL' if str(m).startswith('StablePrompt-PPO')
            else ('RAE' if m == 'AlgPrompt' else m)
        )
        # 优先按 (dataset, seed) 对齐；若 AlgPrompt 未记录 seed（II/II_re/mmlu 项目），
        # 则降级为仅按 dataset 配对（多 seed 内部先求均值）。
        rae_has_seed = df_family[df_family['method_family'] == 'RAE']['seed'].notna().any()
        if rae_has_seed:
            pivot = df_family.pivot_table(index=['dataset', 'seed'],
                                          columns='method_family',
                                          values='final_score', aggfunc='mean')
        else:
            pivot = df_family.pivot_table(index='dataset',
                                          columns='method_family',
                                          values='final_score', aggfunc='mean')
        if 'RL' in pivot.columns and 'RAE' in pivot.columns:
            valid_pairs = pivot.dropna(subset=['RL', 'RAE'])
            n = len(valid_pairs)
            if n >= 1:
                scores_rl = valid_pairs['RL'].values
                scores_rae = valid_pairs['RAE'].values
                rl_mean = float(np.mean(scores_rl))
                rae_mean = float(np.mean(scores_rae))
                mean_diff = rl_mean - rae_mean

                rl_bold = r"\textbf{" + f"{rl_mean:.1f}" + "}" if rl_mean > rae_mean else f"{rl_mean:.1f}"
                rae_bold = r"\textbf{" + f"{rae_mean:.1f}" + "}" if rae_mean > rl_mean else f"{rae_mean:.1f}"

                if n >= MIN_PAIRS:
                    stats_res = calculate_paired_statistics(scores_rl, scores_rae)
                    ci_str = f"[{stats_res['ci_95'][0]:.2f}, {stats_res['ci_95'][1]:.2f}]"
                    cohens_d = stats_res['cohens_d']
                    ci_lower = stats_res['ci_95'][0]
                    ci_upper = stats_res['ci_95'][1]
                else:
                    ci_str = f"-- ($n={n}$)"
                    cohens_d = np.nan
                    ci_lower = np.nan
                    ci_upper = np.nan

                cohens_str = f"{cohens_d:.2f}" if pd.notna(cohens_d) else "--"
                latex_str += f"{family_name} & {rl_bold} & {rae_bold} & {mean_diff:+.2f} & {ci_str} & {cohens_str} \\\\\n"
                
                csv_data.append({
                    'Task Family': family_name,
                    'N pairs': n,
                    'RL (PPO)': rl_mean,
                    'RAE (Ours)': rae_mean,
                    'Delta(RL-RAE)': mean_diff,
                    '95% CI Lower': ci_lower,
                    '95% CI Upper': ci_upper,
                    "Cohen's d": cohens_d,
                })

    latex_str += r"""
\bottomrule
\end{tabular}
}
\end{table}
"""
    save_path = os.path.join(TABLE_DIR, 'tab1_ablation_summary.tex')
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(latex_str.strip())
    print(f"[Table Generated] Ablation Summary Table -> {save_path}")
    
    if csv_data:
        pd.DataFrame(csv_data).to_csv(os.path.join(TABLE_DIR, 'tab1_ablation_summary.csv'), index=False)

def generate_qwen3_table(df_clean):
    if df_clean is None or df_clean.empty:
        return
        
    df_qwen = df_clean[df_clean['agent_model'] == 'Qwen3-8B']
    if df_qwen.empty:
        print("[Pending] Qwen3-8B table generation: No Qwen3 data found.")
        return

    datasets = ['HOTPOTQA', 'IFBENCH', 'HOVER', 'PUPA', 'AIME-2025', 'LIVEBENCH-MATH']
    methods = ['Baseline', 'GEPA', 'FSPO']
    
    latex_str = r"""
\begin{table}[ht]
\centering
\caption{Performance on Qwen3-8B across different tasks.}
\label{tab:qwen_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Dataset} & \textbf{Baseline} & \textbf{GEPA} & \textbf{FSPO (Ours)} & \textbf{GEPA/FSPO Cost} \\ \midrule
"""
    csv_data = []
    for ds in datasets:
        df_ds = df_qwen[df_qwen['dataset'] == ds]
        scores = {}
        row_data = {'Dataset': ds}
        for m in methods:
            m_scores = df_ds[df_ds['method'] == m]['final_score']
            scores[m] = f"{m_scores.mean():.1f}" if not m_scores.empty else "--"
            row_data[m] = m_scores.mean() if not m_scores.empty else np.nan
        
        t_gepa = df_ds[df_ds['method'] == 'GEPA']['rollouts'].mean()
        t_fspo = df_ds[df_ds['method'] == 'FSPO']['rollouts'].mean()
        if pd.notna(t_gepa) and pd.notna(t_fspo):
            cost = f"{t_gepa/1000:.1f}K/{t_fspo/1000:.1f}K"
        else:
            cost = "--"
            
        row_data['GEPA_Cost'] = t_gepa
        row_data['FSPO_Cost'] = t_fspo
        csv_data.append(row_data)
        latex_str += f"{ds} & {scores['Baseline']} & {scores['GEPA']} & {scores['FSPO']} & {cost} \\\\\n"

    latex_str += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    save_path = os.path.join(TABLE_DIR, 'tab2_qwen3.tex')
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(latex_str.strip())
    print(f"[Table Generated] Qwen3 Table -> {save_path}")
    
    if csv_data:
        pd.DataFrame(csv_data).to_csv(os.path.join(TABLE_DIR, 'tab2_qwen3.csv'), index=False)

def generate_gpt4_table(df_clean):
    if df_clean is None or df_clean.empty:
        return
        
    df_gpt4 = df_clean[df_clean['agent_model'] == 'GPT-4.1-mini']
    if df_gpt4.empty:
        print("[Pending] GPT-4.1-mini table generation: No GPT-4 data found.")
        return

    datasets = ['HOTPOTQA', 'IFBENCH', 'HOVER', 'PUPA', 'AIME-2025', 'LIVEBENCH-MATH']
    methods = ['Baseline', 'GEPA', 'FSPO']
    
    latex_str = r"""
\begin{table}[ht]
\centering
\caption{Boundary Conditions on GPT-4.1-mini.}
\label{tab:gpt4_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Dataset} & \textbf{Baseline} & \textbf{GEPA} & \textbf{FSPO (Ours)} & \textbf{Observation} \\ \midrule
"""
    csv_data = []
    for ds in datasets:
        df_ds = df_gpt4[df_gpt4['dataset'] == ds]
        scores = {}
        row_data = {'Dataset': ds}
        for m in methods:
            m_scores = df_ds[df_ds['method'] == m]['final_score']
            scores[m] = f"{m_scores.mean():.1f}" if not m_scores.empty else "--"
            row_data[m] = m_scores.mean() if not m_scores.empty else np.nan
        
        # 动态计算绝对增益
        try:
            gain = float(scores['FSPO']) - float(scores['Baseline'])
            obs = f"{gain:+.1f}"
        except:
            gain = np.nan
            obs = "--"
            
        row_data['Delta(FSPO-Baseline)'] = gain
        csv_data.append(row_data)
        
        latex_str += f"{ds} & {scores['Baseline']} & {scores['GEPA']} & {scores['FSPO']} & {obs} \\\\\n"

    latex_str += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    save_path = os.path.join(TABLE_DIR, 'tab3_gpt4.tex')
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(latex_str.strip())
    print(f"[Table Generated] GPT-4 Table -> {save_path}")
    
    if csv_data:
        pd.DataFrame(csv_data).to_csv(os.path.join(TABLE_DIR, 'tab3_gpt4.csv'), index=False)

# ---------------------------------------------------------
# 7. Cost Breakdown & RCEI Calculation (Table 6 & Table 7)
# ---------------------------------------------------------
def generate_cost_and_rcei_tables(df_clean):
    """
    自动生成 Table 6 (Cost-Drop) 和 Table 7 (RCEI) 
    处理了本地模型 API Cost 为 N/A 的情况，并严格应用了你选中的 RCEI 严谨公式。
    """
    if df_clean is None or df_clean.empty:
        return

    # 排除 dataset==UNKNOWN 的行，避免污染均值
    df_clean = df_clean[df_clean['dataset'] != 'UNKNOWN'].copy()
    print("Generating Cost and RCEI Tables...")
    # --- 1. Table 6: Cost Breakdown ---
    cost_latex = r"""
\begin{table}[h]
\centering
\caption{Comprehensive Cost Breakdown (Averaged across tasks).}
\label{tab:cost_drop}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Method} & \textbf{Rollouts} & \textbf{Est. GPU$\cdot$h} & \textbf{Wall-clock} & \textbf{Est. Token Cost} \\ \midrule
"""
    methods = sorted(df_clean['method'].dropna().unique())
    
    cost_csv_data = []
    for m in methods:
        df_m = df_clean[df_clean['method'] == m]
        if df_m.empty: continue
            
        rollouts = df_m['rollouts'].mean()
        rollouts_str = f"{rollouts:,.0f}" if pd.notna(rollouts) and rollouts > 0 else "1"
        
        runtime_h = df_m['runtime_mins'].mean() / 60.0
        wall_clock = rf"$\sim${runtime_h:.1f}h" if pd.notna(runtime_h) and runtime_h > 0 else "--"
        
        # 智能区分本地模型与 API 模型
        is_local = df_m['agent_model'].str.contains('qwen|gemma', case=False, na=False).any()
        is_api = df_m['agent_model'].str.contains('gpt', case=False, na=False).any()
        
        gpu_h = rf"$\sim${runtime_h:.1f}" if is_local and pd.notna(runtime_h) and runtime_h > 0 else "N/A"
        
        if is_api:
            avg_cost = df_m.loc[df_m['agent_model'].str.contains('gpt', case=False, na=False), 'estimated_cost_usd'].mean()
            token_cost = rf"\${avg_cost:.2f}" if pd.notna(avg_cost) else "--"
        else:
            token_cost = "N/A" # 针对本地模型智能标记为 N/A
            
        m_escaped = m.replace('_', r'\_')
        cost_latex += f"{m_escaped} & {rollouts_str} & {gpu_h} & {wall_clock} & {token_cost} \\\\\n"
        
        cost_csv_data.append({
            'Method': m,
            'Rollouts': rollouts,
            'Runtime_mins': df_m['runtime_mins'].mean(),
            'Est_GPU_h': runtime_h if is_local and pd.notna(runtime_h) and runtime_h > 0 else np.nan,
            'Est_Token_Cost_USD': avg_cost if is_api else np.nan
        })
        
    cost_latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    save_path_cost = os.path.join(TABLE_DIR, 'tab6_cost_drop.tex')
    with open(save_path_cost, "w", encoding="utf-8") as f:
        f.write(cost_latex.strip())
    print(f"[Table Generated] Cost Drop Table -> {save_path_cost}")
    
    if cost_csv_data:
        pd.DataFrame(cost_csv_data).to_csv(os.path.join(TABLE_DIR, 'tab6_cost_drop.csv'), index=False)

    # --- 2. Table 7: RCEI Ranking ---
    rcei_latex = r"""
\begin{table}[h]
\centering
\caption{RCEI Ranking Relative to Unoptimized Baseline.}
\label{tab:rcei}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Method ($\mathcal{M}$)} & \textbf{RER} ($\frac{\mathcal{M}-\mathcal{B}}{100\%-\mathcal{B}}$) & \textbf{Normalized Cost} ($T_{\mathcal{M}} / T_{\mathcal{B}}$) & \textbf{RCEI} \\ \midrule
Baseline & -- & 1 & -- \\
"""
    
    rcei_csv_data = []
    # 考察优化方法（除 Baseline / AlgPrompt 之外的一切）
    for m in [m for m in methods if m not in ('Baseline', 'AlgPrompt')]:
        rers, norm_costs = [], []

        # 严格按 (dataset, agent_model) 对齐；基线 = Baseline 或 AlgPrompt（PPO 禁用同代码库变体）
        m_groups = df_clean[df_clean['method'] == m].groupby(['dataset', 'agent_model'])

        for (ds, model), m_data in m_groups:
            base_data = df_clean[(df_clean['dataset'] == ds) &
                                 (df_clean['agent_model'] == model) &
                                 (df_clean['method'].isin(['Baseline', 'AlgPrompt']))]
            if base_data.empty:
                continue

            score_b = base_data['final_score'].mean()
            score_m = m_data['final_score'].mean()
            t_m = m_data['rollouts'].mean()

            if pd.isna(score_b) or pd.isna(score_m) or score_b >= 100 or pd.isna(t_m) or t_m <= 1:
                continue

            # t_b = baseline 在该 (dataset, model) 下的测试集大小（baseline 需要对每条样本 forward 一次）
            t_b = base_data['test_size'].mean()
            if pd.isna(t_b) or t_b <= 0:
                t_b = m_data['test_size'].mean()
            if pd.isna(t_b) or t_b <= 0:
                continue

            rer = (score_m - score_b) / (100.0 - score_b) * 100.0
            norm_cost = t_m / t_b

            if norm_cost > 1:
                rers.append(rer)
                norm_costs.append(norm_cost)
            
        if rers:
            avg_rer = np.mean(rers)
            avg_norm_cost = np.mean(norm_costs)
            # 依据所有数据集的平均 RER 和平均 Normalized Cost 计算整体 RCEI
            avg_rcei = avg_rer / np.log10(avg_norm_cost) if avg_norm_cost > 1 else 0.0
            
            # 使用科学计数法增强大开销方法的展示
            cost_str = rf"$\sim {avg_norm_cost/1000:.1f} \times 10^3$" if avg_norm_cost >= 1000 else rf"$\sim {avg_norm_cost:.0f}$"
            m_escaped = m.replace('_', r'\_')
            m_bold = f"\\textbf{{{m_escaped}}}" if m == 'FSPO' else m_escaped
            rcei_bold = f"\\textbf{{{avg_rcei:.2f}}}" if m == 'FSPO' else f"{avg_rcei:.2f}"
            rer_bold = f"\\textbf{{{avg_rer:+.1f}\\%}}" if m == 'FSPO' else f"{avg_rer:+.1f}\\%"
            
            rcei_latex += f"{m_bold} & {rer_bold} & {cost_str} & {rcei_bold} \\\\\n"
            
            rcei_csv_data.append({
                'Method': m,
                'RER(%)': avg_rer,
                'Normalized_Cost': avg_norm_cost,
                'RCEI': avg_rcei
            })
            
    rcei_latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    save_path_rcei = os.path.join(TABLE_DIR, 'tab7_rcei.tex')
    with open(save_path_rcei, "w", encoding="utf-8") as f:
        f.write(rcei_latex.strip())
    print(f"[Table Generated] RCEI Table -> {save_path_rcei}")
    
    if rcei_csv_data:
        pd.DataFrame(rcei_csv_data).to_csv(os.path.join(TABLE_DIR, 'tab7_rcei.csv'), index=False)

def format_score_with_std(scores):
    """
    处理多个随机种子的得分：
    - 排除明显异常 (例如得分为 0.0 或极低)
    - 多个 seed 返回 Mean(±Std)
    - 单个 seed 返回 Max
    """
    valid_scores = scores.dropna()
    valid_scores = valid_scores[valid_scores > 1.0] # 排除明显异常的失败 run
    if valid_scores.empty:
        return "--", np.nan
    if len(valid_scores) == 1:
        return f"{valid_scores.max():.1f}", valid_scores.max()
    
    mean_val = valid_scores.mean()
    std_val = valid_scores.std()
    if std_val > 0.05:
        return rf"{mean_val:.1f}($\pm${std_val:.1f})", mean_val
    return f"{valid_scores.max():.1f}", valid_scores.max()

# ---------------------------------------------------------
# 8. Full Comprehensive Table (Table 8)
# ---------------------------------------------------------
def generate_full_comprehensive_table(df_clean):
    """
    自动生成附录中的完整大表格，包含 Input Token, Output Token, Rollouts, RER, Normalized Cost, RCEI 等
    """
    if df_clean is None or df_clean.empty:
        return
        
    print("Generating Full Comprehensive Table...")
    latex_str = r"""
{\scriptsize
\setlength{\tabcolsep}{6pt}
\begin{longtable}{@{}p{5.5cm}p{4.5cm}p{2.5cm}lcrrrcc@{}}
\caption{Full Comprehensive Results across all datasets and methods.}
\label{tab:full_comprehensive} \\
\toprule
\textbf{Dataset} & \textbf{Model} & \textbf{Method} & \textbf{Score} & \textbf{Rollouts} & \textbf{In Toks} & \textbf{Out Toks} & \textbf{RER} & \textbf{Norm Cost} & \textbf{RCEI} \\ \midrule
\endfirsthead
\caption[]{Full Comprehensive Results (Continued)} \\
\toprule
\textbf{Dataset} & \textbf{Model} & \textbf{Method} & \textbf{Score} & \textbf{Rollouts} & \textbf{In Toks} & \textbf{Out Toks} & \textbf{RER} & \textbf{Norm Cost} & \textbf{RCEI} \\ \midrule
\endhead
\bottomrule
\endfoot
\endlastfoot
"""
    
    csv_data = []
    # 按照数据集和目标代理模型进行分组
    groups = df_clean.groupby(['dataset', 'agent_model'])
    
    for (ds, model), group in groups:
        # 提取基线分数作为对照（Baseline 或 AlgPrompt = PPO 未启用的同代码库变体）
        base_data = group[group['method'].isin(['Baseline', 'AlgPrompt'])]
        score_b = base_data['final_score'].mean() if not base_data.empty else np.nan
        
        # 计算每一组中各个 Method 的情况
        method_stats = []
        for m in group['method'].unique():
            m_data = group[group['method'] == m]
            
            # 调用多 Seed 格式化工具
            score_str, score_m = format_score_with_std(m_data['final_score'])
            rollouts = m_data['rollouts'].mean()
            in_toks = m_data['input_tokens'].mean()
            out_toks = m_data['output_tokens'].mean()
            
            rer_str = "--"
            norm_cost_str = "--"
            rcei_str = "--"
            raw_rer = np.nan
            raw_norm_cost = np.nan
            raw_rcei = np.nan
            
            if pd.notna(score_b) and score_b < 100 and pd.notna(score_m):
                rer = (score_m - score_b) / (100.0 - score_b) * 100.0
                if m in base_data['method'].values:
                    rer_str = "--"
                    norm_cost_str = "1"
                    rcei_str = "--"
                    raw_norm_cost = 1.0
                elif pd.notna(rollouts) and rollouts > 1:
                    t_b = 1.0
                    norm_cost = rollouts / t_b
                    rcei = rer / np.log10(norm_cost) if norm_cost > 1 else 0.0
                    
                    rer_str = rf"{rer:+.1f}\%"
                    norm_cost_str = rf"{norm_cost:.0f}"
                    rcei_str = rf"{rcei:.2f}"
                    raw_rer = rer
                    raw_norm_cost = norm_cost
                    raw_rcei = rcei
            
            rollouts_str = f"{rollouts:.0f}" if pd.notna(rollouts) else "--"
            
            def format_tokens(toks):
                if pd.isna(toks) or toks == 0:
                    return "--"
                if toks >= 1e6:
                    return f"{toks/1e6:.2f}M"
                elif toks >= 1e3:
                    return f"{toks/1e3:.1f}K"
                else:
                    return f"{toks:.0f}"
            
            in_toks_str = format_tokens(in_toks)
            out_toks_str = format_tokens(out_toks)
            
            m_escaped = m.replace('_', r'\_')
            method_stats.append((ds, model, m_escaped, score_str, rollouts_str, in_toks_str, out_toks_str, rer_str, norm_cost_str, rcei_str))
            
            csv_data.append({
                'Dataset': ds,
                'Model': model,
                'Method': m,
                'Score_Mean': score_m,
                'Score_Std': m_data['final_score'].std() if len(m_data['final_score'].dropna()) > 1 else np.nan,
                'Rollouts': rollouts,
                'In_Tokens': in_toks,
                'Out_Tokens': out_toks,
                'RER(%)': raw_rer,
                'Norm_Cost': raw_norm_cost,
                'RCEI': raw_rcei
            })
        
        # 对方法排序，确保 Baseline 在分组的顶部显示
        def sort_key(x):
            if 'Baseline' in x[2] or r'cs\_0.0' in x[2]:
                return 0
            return 1
            
        method_stats.sort(key=sort_key)
        
        for idx, row_data in enumerate(method_stats):
            ds_escaped = str(row_data[0]).replace('_', r'\_')
            md_escaped = str(row_data[1]).replace('_', r'\_')
            latex_str += f"{ds_escaped} & {md_escaped} & {row_data[2]} & {row_data[3]} & {row_data[4]} & {row_data[5]} & {row_data[6]} & {row_data[7]} & {row_data[8]} & {row_data[9]} \\\\\n"
        
        latex_str += "\\midrule\n"
        
    # 清理末尾多余的下划线分隔
    if latex_str.endswith("\\midrule\n"):
        latex_str = latex_str[:-9]
    
    latex_str += r"""\bottomrule
\end{longtable}
}
"""
    save_path = os.path.join(TABLE_DIR, 'tab8_full_comprehensive.tex')
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(latex_str.strip())
    print(f"[Table Generated] Full Comprehensive Table -> {save_path}")
    
    if csv_data:
        pd.DataFrame(csv_data).to_csv(os.path.join(TABLE_DIR, 'tab8_full_comprehensive.csv'), index=False)

if __name__ == "__main__":
    print("=== Starting Data Processing in 'results' ===")
    df_cleaned = load_and_clean_data()
    
    # 排除明显异常与检查 PPO 配置
    # 1. 排除得分为0或NaN的异常数据
    df_cleaned = df_cleaned[df_cleaned['final_score'] > 0]
    # 2. 检查 StablePrompt (PPO) 实验的有效性：如果有 PPO 方法，则其 rollouts 应该显著大于 1，否则为异常/失败 run
    is_ppo = df_cleaned['method'].str.contains('StablePrompt-PPO', case=False, na=False)
    invalid_ppo = is_ppo & (df_cleaned['rollouts'] <= 1)
    df_cleaned = df_cleaned[~invalid_ppo]
    
    plot_best_of_n_scaling(df_cleaned)
    plot_pareto_frontier(df_cleaned)
    generate_ablation_latex_table(df_cleaned)
    generate_qwen3_table(df_cleaned)
    generate_gpt4_table(df_cleaned)
    generate_cost_and_rcei_tables(df_cleaned)
    generate_full_comprehensive_table(df_cleaned)
    print("=== All tasks completed successfully ===")