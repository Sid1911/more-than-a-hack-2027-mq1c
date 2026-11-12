@echo off
rem Double-click after run.bat: shows our submission as a readable list of possessions,
rem with the goals from the organisers' automatic tags.
setlocal
set "HERE=%~dp0"
set /p TEAM=<"%HERE%team.txt"
set "PY=%HERE%.venv\Scripts\python.exe"
set "SUB=%HERE%submission\%TEAM%.csv"
set "TAGS=%HERE%data\tactical\BAR-ATH_tactical_smart_tagging.xml"
if not exist "%TAGS%" set "TAGS=%HERE%data\BAR-ATH_tactical_smart_tagging.xml"
if not exist "%TAGS%" set "TAGS=%HERE%..\BAR-ATH_tactical_smart_tagging.xml"
if not exist "%PY%" (
  echo Run run.bat first - it sets up Python and makes the submission.
  pause
  exit /b 1
)
if not exist "%SUB%" (
  echo No submission yet at %SUB% - run run.bat first.
  pause
  exit /b 1
)
"%PY%" "%HERE%scripts\summary.py" "%SUB%" --tagging "%TAGS%"
echo.
echo The full list is in submission\%TEAM%_possessions.csv - open it in Excel.
pause
