@echo off
REM MeeTARA Lab - Simple Voice & Translation Testing Batch Script
REM Run this script to test your voice and translation models

echo ============================================================
echo MeeTARA Lab - Voice & Translation Testing
echo ============================================================

REM Check if we're in the right directory
if not exist "models\production" (
    echo ERROR: Please run this script from the meetara-lab root directory
    pause
    exit /b 1
)

echo.
echo Checking GGUF Model...
echo ----------------------

REM Check if GGUF model exists
set "ggufPath=models\production\B_universal\meetara-qwen2.5-7B-instruct-Q4_K_M-20250809.gguf"
if exist "%ggufPath%" (
    echo OK: Found GGUF model
    for %%A in ("%ggufPath%") do echo    Size: %%~zA bytes
) else (
    echo ERROR: GGUF model not found: %ggufPath%
)

echo.
echo Checking Translation Models...
echo -----------------------------

REM Check Hindi model
set "hiPath=models\production\translation_models\translation_bundle_20250809_222911\hi_model"
if exist "%hiPath%" (
    echo OK: Hindi model found
) else (
    echo ERROR: Hindi model not found
)

REM Check Telugu model
set "tePath=models\production\translation_models\translation_bundle_20250809_222911\te_model"
if exist "%tePath%" (
    echo OK: Telugu model found
) else (
    echo ERROR: Telugu model not found
)

echo.
echo Checking Speech Models...
echo ------------------------

REM Check speech models directory
set "speechPath=models\production\speech_models"
if exist "%speechPath%" (
    echo OK: Speech models directory found
    
    REM Check subdirectories
    if exist "%speechPath%\emotion" echo    OK: emotion subdirectory
    if exist "%speechPath%\voice" echo    OK: voice subdirectory
    if exist "%speechPath%\routing" echo    OK: routing subdirectory
) else (
    echo ERROR: Speech models directory not found
)

echo.
echo Checking llama.cpp...
echo --------------------

REM Check llama.cpp build
set "llamaPath=llama.cpp\build\bin\llama-cli.exe"
if exist "%llamaPath%" (
    echo OK: llama.cpp is built and ready
) else (
    echo WARNING: llama-cli.exe not found
    echo    Please build: cd llama.cpp ^&^& mkdir build ^&^& cd build ^&^& cmake .. ^&^& cmake --build . --config Release
)

echo.
echo Checking Python...
echo -----------------

REM Check Python
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo OK: Python is available
    python --version
) else (
    echo ERROR: Python not available
)

echo.
echo ============================================================
echo Testing completed!
echo ============================================================
pause
