$env:VIRTUAL_ENV='c:\Users\123\Desktop\gepa\gepa-artifact\.venv-artifact'
$env:PATH='c:\Users\123\Desktop\gepa\gepa-artifact\.venv-artifact\Scripts;' + $env:PATH
python gepa-artifact/scripts/run_hotpotqa_openrouter_gpt41mini.py --optimizer Baseline --num-threads 16
