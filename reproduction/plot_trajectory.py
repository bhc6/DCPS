import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize wandb API
api = wandb.Api()

print("Fetching run 92emydsq (GFB)...")
r1 = api.run("awesome-prompt/GFB_TC/92emydsq")
df1 = r1.history(samples=1000)

print("Fetching run hxxqjxon (PPO)...")
r2 = api.run("awesome-prompt/algprompt_classification_sst2/hxxqjxon")
df2 = r2.history(samples=1000)

# -------------------------------------------------------------
# 1. Process GFB (df1) - Epoch-by-epoch (0 to 100)
# -------------------------------------------------------------
df1_val = df1.dropna(subset=['epoch_summary/mean_accuracy']).copy()
df1_val['epoch'] = df1_val['epoch'].astype(int)
df1_val = df1_val.sort_values('epoch')

gfb_epochs = np.arange(101)
gfb_mean_accs = df1_val['epoch_summary/mean_accuracy'].values

gfb_running_best_full = []
current_best = 0.0
for acc in gfb_mean_accs:
    current_best = max(current_best, acc)
    gfb_running_best_full.append(current_best)

gfb_running_best = np.zeros(101)
for t in range(100):
    gfb_running_best[t] = gfb_running_best_full[t] * 100.0
gfb_running_best[100] = gfb_running_best_full[99] * 100.0
gfb_final_test = df1['final/best_accuracy'].dropna().values[0] * 100.0

# -------------------------------------------------------------
# 2. Process PPO (df2) - Epoch-by-epoch (0 to 100)
# -------------------------------------------------------------
df2_val = df2.dropna(subset=['epoch_summary/mean_accuracy']).copy()
df2_val['epoch'] = df2_val['epoch'].astype(int)
df2_val = df2_val.sort_values('epoch')

ppo_epochs = np.arange(101)
ppo_mean_accs = df2_val['epoch_summary/mean_accuracy'].values

ppo_running_best_full = []
current_best = 0.0
for acc in ppo_mean_accs:
    current_best = max(current_best, acc)
    ppo_running_best_full.append(current_best)

ppo_running_best = np.zeros(101)
for t in range(100):
    ppo_running_best[t] = ppo_running_best_full[t] * 100.0
ppo_running_best[100] = ppo_running_best_full[99] * 100.0
ppo_final_test = df2['final/best_accuracy'].dropna().values[0] * 100.0

# -------------------------------------------------------------
# 3. Plotting
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=180)
fig.patch.set_alpha(0)

# Colors
gfb_color = '#2b5c8f'       # Deep blue
ppo_color = '#d95f02'       # Orange

# Plot running best curves (epoch-by-epoch)
ax.plot(gfb_epochs, gfb_running_best, label='StablePrompt-DCPS (Val Best)', color=gfb_color, linewidth=1.4, zorder=2)
ax.plot(ppo_epochs, ppo_running_best, label='StablePrompt-PPO (Val Best)', color=ppo_color, linewidth=1.4, linestyle='--', zorder=2)

# Plot final test accuracy points at epoch 100 using stars
ax.scatter([100], [gfb_final_test], color=gfb_color, marker='*', s=86, edgecolor='black', linewidth=0.25, alpha=0.88, zorder=4, label='StablePrompt-DCPS (Final Test)')
ax.scatter([100], [ppo_final_test], color=ppo_color, marker='*', s=86, edgecolor='black', linewidth=0.25, alpha=0.88, zorder=4, label='StablePrompt-PPO (Final Test)')

# Labels (in-plot title dropped; the LaTeX sub-caption carries model/dataset).
# Fonts enlarged so text stays legible after down-scaling to ~6cm (LNCS 0.49\textwidth panel).
ax.set_xlabel("Epochs", fontsize=15, fontweight='bold')
ax.set_ylabel("Accuracy (%)", fontsize=15, fontweight='bold')
ax.tick_params(axis='both', labelsize=13)
ax.set_xlim(left=-5, right=105)
ax.set_ylim(bottom=50, top=100) # mean_accuracy values are lower, starting around 50-70%

# Grid customization
ax.grid(alpha=0.25)
ax.set_facecolor("none")

# Legend
ax.legend(loc='lower right', fontsize=12, framealpha=0.9, facecolor='white')

# Save
fig.tight_layout()
fig.savefig("results/figures/stableprompt_trajectory_sst2.png", bbox_inches="tight", transparent=True, dpi=300)
fig.savefig("results/figures/stableprompt_trajectory_sst2.pdf", bbox_inches="tight", transparent=True)
print("Successfully generated stableprompt_trajectory_sst2.png and stableprompt_trajectory_sst2.pdf with StablePrompt-DCPS vs StablePrompt-PPO.")
