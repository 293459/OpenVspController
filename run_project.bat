@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0scripts\run_project.ps1" %*
exit /b %ERRORLEVEL%
