@echo off
REM Start SpatialAI.
REM   start_spatialai.bat        development: backend on :8000 + Vite on :5173 (two windows)
REM   start_spatialai.bat prod   builds the frontend once and serves everything from :8000
cd /d "%~dp0"
if not exist "backend\.venv\Scripts\python.exe" (
  echo SpatialAI is not installed yet. Run install_windows.bat first.
  pause
  exit /b 1
)

REM Load .env (KEY=VALUE lines, # comments ignored)
if exist ".env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do if not "%%B"=="" set "%%A=%%B"
)

if /i "%1"=="prod" goto prod

echo Starting backend on http://127.0.0.1:8000 ...
start "SpatialAI backend" cmd /k "cd /d %~dp0backend && .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
echo Starting frontend on http://localhost:5173 ...
start "SpatialAI frontend" cmd /k "cd /d %~dp0frontend && npm run dev"
ping 127.0.0.1 -n 7 >nul
start "" http://localhost:5173
echo.
echo SpatialAI is running. Close the two terminal windows to stop it.
exit /b 0

:prod
echo Building frontend ...
pushd frontend
call npm run build || (popd & echo Frontend build failed. & pause & exit /b 1)
popd
echo Serving SpatialAI on http://127.0.0.1:8000
start "" http://127.0.0.1:8000
cd backend
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
