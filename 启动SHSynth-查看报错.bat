@echo off
rem ===================================================================
rem  SHSynth GUI launcher (keeps the console open to show errors)
rem  Use this one when the GUI does not start, so the traceback stays
rem  visible instead of vanishing with the console window.
rem
rem  Keep this file ASCII-only (see the note in the normal launcher).
rem ===================================================================
setlocal
pushd "%~dp0"

set "PY=C:\Users\pengzhenran\anaconda3\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================================
echo  SHSynth GUI - debug launcher
echo  Python: %PY%
echo ============================================================
"%PY%" -m shsynth.gui %*

echo.
echo ------------------------------------------------------------
echo  The GUI has exited (or failed).  The message above is the
echo  reason.  Press any key to close this window.
echo ------------------------------------------------------------
pause >nul

popd
endlocal
