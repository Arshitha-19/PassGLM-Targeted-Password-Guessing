@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv was not found.
    echo This ZIP contains the project, not your local virtual environment.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo.
echo [1/4] Generating synthetic dataset...
python src\generate_dataset.py
if errorlevel 1 exit /b 1

echo.
echo [2/4] Creating train/test split...
python src\prepare_dataset.py
if errorlevel 1 exit /b 1

echo.
echo [3/4] Creating model-specific datasets...
python src\prepare_model_data.py
if errorlevel 1 exit /b 1

echo.
echo [4/4] Running three-model experiment...
python src\run_experiments.py
if errorlevel 1 exit /b 1

echo.
echo ============================================
echo PIPELINE COMPLETED
echo ============================================
echo Metrics: results\metrics\
echo Predictions: results\predictions\
pause
