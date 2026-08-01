@echo off
setlocal

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py -3 "%~dp0run.py" %*
) else (
    python "%~dp0run.py" %*
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo Something went wrong. See the messages above.
    pause
)
