@echo off
REM SpatialAI installer for Windows 11. Pass options through, e.g.:  install_windows.bat -Cuda cu126
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_environment.ps1" %*
if errorlevel 1 (
  echo.
  echo Installation did not finish. Read the message above, fix it, then run install_windows.bat again.
  pause
  exit /b 1
)
pause
