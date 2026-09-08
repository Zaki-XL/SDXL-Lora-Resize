@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ========================================================
echo   SDXL Model Quantizer - 仮想環境セットアップ
echo ========================================================
echo.

cd /d "%~dp0"

python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [エラー] Python が見つかりませんでした。Python 3.10以上をインストールしてください。
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [情報] 仮想環境 (.venv) を作成しています...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [エラー] 仮想環境の作成に失敗しました。
        pause
        exit /b 1
    )
)

echo [情報] 仮想環境をアクティベートしています...
call .venv\Scripts\activate.bat

echo [情報] pip を更新しています...
python -m pip install --upgrade pip

echo [情報] 必要なパッケージをインストールしています...
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

echo.
echo [情報] ランチャー (SDXL_Quantizer.exe) を生成しています...
if exist "build_launcher.bat" (
    call build_launcher.bat
)

echo.
echo ========================================================
echo [成功] 環境構築が完了しました！
echo SDXL_Quantizer.exe または run.bat から起動できます。
echo (※設定ファイル config.ini は初回起動時に自動生成されます)
echo ========================================================
pause
