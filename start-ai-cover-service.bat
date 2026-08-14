@echo off
setlocal
cd /d "%~dp0"

set "AI_COVER_HOST=0.0.0.0"
set "AI_COVER_PORT=18888"
set "AI_COVER_MAX_UPLOAD_MB=200"
set "AI_COVER_MAX_AUDIO_SECONDS=900"

runtime\python.exe tools\ai_cover_service.py
endlocal
