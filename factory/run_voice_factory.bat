@echo off
echo ========================================
echo MeeTARA Voice Service Factory
echo ========================================
echo.
echo Creating voice domain-specific PKL files...
echo Output: services/speech/voice/
echo.

python factory/voice_service_factory.py

echo.
echo ========================================
echo Factory execution complete!
echo ========================================
pause
