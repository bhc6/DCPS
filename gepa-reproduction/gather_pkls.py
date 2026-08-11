import os
import shutil
from pathlib import Path

base_dir = Path("c:/Users/123/Desktop/gepa-rp/seed_0")
output_dir = Path("c:/Users/123/Desktop/gepa-rp/final_pkls")
output_dir.mkdir(exist_ok=True, parents=True)

copied_count = 0

for exp_dir in base_dir.iterdir():
    if exp_dir.is_dir() and exp_dir.name != "final_pkls":
        pkl_path = exp_dir / "evaluation_results" / "optimized_program" / "program.pkl"
        if pkl_path.exists():
            dest_name = f"{exp_dir.name}.pkl"
            dest_path = output_dir / dest_name
            shutil.copy2(pkl_path, dest_path)
            print(f"Copied {exp_dir.name} pkl.")
            copied_count += 1

print(f"Total copied: {copied_count}")
