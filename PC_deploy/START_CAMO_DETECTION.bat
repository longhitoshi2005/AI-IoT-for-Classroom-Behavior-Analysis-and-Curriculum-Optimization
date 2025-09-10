@echo off
echo =======================================
echo Camo + iPhone Detection Setup
echo =======================================
echo.
echo 1. Testing Camo connection...
echo.

python test_camo.py

echo.
echo =======================================
echo If test successful, run detection with:
echo python deploy_video_detection.py --input [CAMERA_ID]
echo.
echo Example:
echo python deploy_video_detection.py --input 0
echo python deploy_video_detection.py --input 1
echo =======================================
pause
