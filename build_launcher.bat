@echo off
chcp 65001 > nul
cd /d "%~dp0"

set CSC=C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe
if not exist "%CSC%" (
    set CSC=C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe
)

if not exist "%CSC%" (
    echo [ERROR] csc.exe not found.
    pause
    exit /b 1
)

set ICON_OPT=
if exist "assets\app_icon.ico" (
    set ICON_OPT=/win32icon:assets\app_icon.ico
)

echo [INFO] Compiling SDXL_Quantizer.exe with icon...
"%CSC%" /target:winexe /out:SDXL_Quantizer.exe %ICON_OPT% /r:System.Windows.Forms.dll /r:System.dll Launcher.cs

if %errorlevel% equ 0 (
    echo [SUCCESS] SDXL_Quantizer.exe generated successfully!
) else (
    echo [ERROR] Compilation failed.
)
