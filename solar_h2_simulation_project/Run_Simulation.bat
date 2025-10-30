@echo off
title Solar-Hydrogen Simulation Launcher
echo ============================================
echo   Launching Solar–Hydrogen Simulation App...
echo ============================================

:: Go to the correct project directory
cd /d "C:\Users\Sohel pc\Downloads\Solar-Hydrogen simulation\solar_h2_simulation_project"

:: Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate the environment
call venv\Scripts\activate

:: Install dependencies (only if needed)
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo requirements.txt not found! Please ensure it exists in this folder.
)

:: Run Streamlit app
if exist solar_h2_simulation.py (
    streamlit run solar_h2_simulation.py
) else (
    echo solar_h2_simulation.py not found! Please check file name.
)

pause
