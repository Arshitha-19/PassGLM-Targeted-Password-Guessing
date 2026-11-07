@echo off
setlocal
cd /d "%~dp0"
call .venv\Scripts\activate.bat

echo ============================================
echo TOP-100 RE-EVALUATION - NO TRAINING
echo ============================================

python src\evaluate.py --mode pii --beams 100
if errorlevel 1 exit /b 1

python src\evaluate.py --mode old --beams 100
if errorlevel 1 exit /b 1

python src\evaluate.py --mode combined --beams 100
if errorlevel 1 exit /b 1

echo.
echo ============================================
echo COMPLETED
echo ============================================
echo Results:
echo results\metrics\pii_top100.json
echo results\metrics\old_top100.json
echo results\metrics\combined_top100.json
pause
