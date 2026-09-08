@echo off
chcp 65001 > nul
cd /d "%~dp0"

if not exist ".venv" (
    echo [情報] .venv が見つかりません。setup_env.bat を実行します...
    call setup_env.bat
)

call .venv\Scripts\activate.bat
start "" .venv\Scripts\pythonw.exe src\main.py
