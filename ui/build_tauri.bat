@echo off
cd /d "%~dp0"
set PATH=C:\Users\dzume\.cargo\bin;C:\Program Files\nodejs;%PATH%
call npx tauri build
echo EXIT_CODE=%ERRORLEVEL%
