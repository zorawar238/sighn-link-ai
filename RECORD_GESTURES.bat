@echo off
title ISL Gesture Recorder
color 0A
echo ============================================
echo    ISL GESTURE DATA COLLECTOR
echo ============================================
echo.
echo  A camera window will open.
echo  Follow the instructions on screen.
echo  Press SPACE to skip, Q to quit safely.
echo.
echo ============================================
cd /d "c:\Users\nisha\OneDrive\Documents\antigravity\WEATHER APP\isl-translator-ai"
.venv\Scripts\python scripts/collect_real_data.py
echo.
echo ============================================
echo   DONE! Come back to the chat now.
echo ============================================
pause
