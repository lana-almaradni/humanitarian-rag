@echo off
echo Starting Humanitarian Policy Assistant...
cd /d "%~dp0"
call venv\Scripts\activate.bat
streamlit run interface.py
pause