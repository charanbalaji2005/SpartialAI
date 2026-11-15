@echo off
REM Run training commands inside the SpatialAI environment, e.g.
REM   train.bat training.datasets.download coco --parts val annotations
REM   train.bat training.trainers.train_depth --config depth.yaml
REM   train.bat training.train_all --quick
cd /d "%~dp0"
if not exist "backend\.venv\Scripts\python.exe" (
  echo Run install_windows.bat -Training first.
  exit /b 1
)
backend\.venv\Scripts\python.exe -m %*
