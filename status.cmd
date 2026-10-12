@echo off
rem Show per-capture run status (done / frames predicted so far / missing) and GPU use.
"%~dp0.venv\Scripts\python.exe" "%~dp0status.py" %*
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>nul
