@echo off
echo ========================================================
echo  Launching NeuroCell GAN Modern Web Application...
echo ========================================================
echo.
python -m uvicorn app:app --host 0.0.0.0 --port 8000
pause
