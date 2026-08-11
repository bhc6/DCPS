import pandas as pd
import re

file_path = 'GFBablationResult.xlsx'
xl = pd.ExcelFile(file_path)

latex_content = "\\section{Ablation Study Results}\n"
latex_content += "The detailed ablation experimental results across multiple datasets are summarized in the following tables.\n\n"

for sheet in xl.sheet_names:
    df = xl.parse(sheet)
    df = df.fillna('-')
    
    # Escape LaTeX special characters in columns
    df.columns = [str(c).replace('_', '\\_').replace('%', '\\%') for c in df.columns]
    
    # Convert data to string and escape _
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x).replace('_', '\\_').replace('%', '\\%') if isinstance(x, str) else x)
    
    # Generate tabular content
    table_tex = df.to_latex(index=False, escape=False) # pandas 2.0+ handles basic escapes if escape=True, but we manually handled _ and %. escape=False avoids double escaping.
    
    # Remove \begin{table} and \end{table} from pandas output to wrap it with resizebox
    table_tex = table_tex.replace('\\begin{table}\n', '').replace('\\end{table}\n', '')
    
    latex_content += "\\begin{table}[hbt!]\n"
    latex_content += f"\\caption{{Ablation results on {sheet} dataset.}}\\label{{tab:{sheet.replace(' ', '_')}}}\n"
    latex_content += "\\centering\n"
    latex_content += "\\resizebox{\\textwidth}{!}{\n"
    latex_content += table_tex
    latex_content += "}\n\\end{table}\n\n"

with open('samplepaper.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

# Replace from \section{First Section} to just before \begin{credits}
pattern = re.compile(r'\\section\{First Section\}.*?(?=\\begin\{credits\})', re.DOTALL)
new_tex = pattern.sub(lambda m: latex_content, tex)

with open('samplepaper.tex', 'w', encoding='utf-8') as f:
    f.write(new_tex)

print("Latex tables injected successfully.")
