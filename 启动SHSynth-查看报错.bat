@echo off
rem ===================================================================
rem  SHSynth GUI launcher, WITH console.
rem  Use this one when the GUI does not start: it keeps the window open
rem  and prints the traceback.
rem
rem  Keep this file ASCII-only: cmd.exe reads .bat in the OEM code page
rem  (GBK on a Chinese Windows) and would mis-parse UTF-8 Chinese here.
rem  Interpreter lookup order: SHSYNTH_PYTHON, Anaconda under %USERPROFILE%,
rem  the py launcher, then python on PATH.
rem ===================================================================
setlocal
pushd "%~dp0"

set "PY="
if defined SHSYNTH_PYTHON if exist "%SHSYNTH_PYTHON%" set "PY=%SHSYNTH_PYTHON%"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python" for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do if not defined PY if exist "%%~fD\python.exe" set "PY=%%~fD\python.exe"

echo ============================================================
echo  SHSynth GUI - debug launcher
if defined PY (
  echo  Python: %PY%
) else (
  echo  Python: py -3  ^(from PATH^)
)
echo ============================================================
echo.

if defined PY (
  "%PY%" -m shsynth.gui %*
) else (
  py -3 -m shsynth.gui %*
)

echo.
echo ------------------------------------------------------------
echo  The GUI has exited (or failed).  The message above is the
echo  reason.  Press any key to close this window.
echo ------------------------------------------------------------
pause >nul

popd
endlocal
