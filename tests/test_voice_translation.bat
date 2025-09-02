@echo off
REM MeeTARA Lab - Voice & Translation Testing Batch Script
REM Run this script to test your voice and translation models

echo 🚀 MeeTARA Lab - Voice ^& Translation Testing
echo ============================================================

REM Check if we're in the right directory
if not exist "models\production" (
    echo ❌ Error: Please run this script from the meetara-lab root directory
    pause
    exit /b 1
)

echo.
echo 🔨 Checking llama.cpp build...
if not exist "llama.cpp\build\bin\llama-cli.exe" (
    echo ⚠️ llama-cli.exe not found. Please build llama.cpp first.
    echo    Run: cd llama.cpp ^&^& mkdir build ^&^& cd build ^&^& cmake .. ^&^& cmake --build . --config Release
    echo.
) else (
    echo ✅ llama.cpp is built and ready
)

echo.
echo 🤖 Testing GGUF Model...
set GGUF_PATH=models\production\B_universal\meetara-qwen2.5-7B-instruct-Q4_K_M-20250809.gguf
if exist "%GGUF_PATH%" (
    echo ✅ Found GGUF model: %GGUF_PATH%
    for %%A in ("%GGUF_PATH%") do (
        set /a SIZE=%%~zA/1024/1024/1024
        echo ✅ Model size: !SIZE! GB
    )
) else (
    echo ❌ GGUF model not found: %GGUF_PATH%
)

echo.
echo 🌐 Testing Translation Models...
set HI_PATH=models\production\translation_models\translation_bundle_20250809_222911\hi_model
set TE_PATH=models\production\translation_models\translation_bundle_20250809_222911\te_model

if exist "%HI_PATH%" (
    echo ✅ Hindi model found: %HI_PATH%
    if exist "%HI_PATH%\model.pt" (
        for %%A in ("%HI_PATH%\model.pt") do (
            set /a SIZE=%%~zA/1024/1024
            echo    Model size: !SIZE! MB
        )
    )
) else (
    echo ❌ Hindi model not found
)

if exist "%TE_PATH%" (
    echo ✅ Telugu model found: %TE_PATH%
    if exist "%TE_PATH%\model.pt" (
        for %%A in ("%TE_PATH%\model.pt") do (
            set /a SIZE=%%~zA/1024/1024
            echo    Model size: !SIZE! MB
        )
    )
) else (
    echo ❌ Telugu model not found
)

echo.
echo 🎤 Testing Speech Models...
set SPEECH_PATH=models\production\speech_models
if exist "%SPEECH_PATH%" (
    echo ✅ Speech models directory found: %SPEECH_PATH%
    
    if exist "%SPEECH_PATH%\emotion" echo    ✅ emotion: %SPEECH_PATH%\emotion
    if exist "%SPEECH_PATH%\voice" echo    ✅ voice: %SPEECH_PATH%\voice
    if exist "%SPEECH_PATH%\routing" echo    ✅ routing: %SPEECH_PATH%\routing
) else (
    echo ❌ Speech models directory not found
)

echo.
echo 🐍 Testing Python Components...
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Python available
    if exist "tests\quick_test.py" (
        echo 🔄 Running Python quick test...
        python tests\quick_test.py
        if %ERRORLEVEL% EQU 0 (
            echo ✅ Python quick test completed successfully
        ) else (
            echo ❌ Python quick test failed
        )
    ) else (
        echo ⚠️ Quick test script not found
    )
) else (
    echo ❌ Python not available
)

echo.
echo ============================================================
echo 📊 TEST SUMMARY
echo ============================================================

REM Count available components
set /a AVAILABLE=0
set /a TOTAL=5

if exist "llama.cpp\build\bin\llama-cli.exe" set /a AVAILABLE+=1
if exist "%GGUF_PATH%" set /a AVAILABLE+=1
if exist "%HI_PATH%" set /a AVAILABLE+=1
if exist "%SPEECH_PATH%" set /a AVAILABLE+=1
python --version >nul 2>&1 && set /a AVAILABLE+=1

echo Overall: !AVAILABLE!/!TOTAL! components available

if !AVAILABLE! EQU !TOTAL! (
    echo 🎉 All components available! Your MeeTARA Lab is ready for testing.
) else (
    echo ⚠️ Some components are missing. Please check the issues above.
)

echo.
echo 💡 Next steps:
echo    1. Run PowerShell test: .\tests\test_voice_translation.ps1
echo    2. Run Python test: python tests\voice_translation_test.py
echo    3. Check testing guide: tests\TESTING_GUIDE.md
echo.

pause
