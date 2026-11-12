@echo off
rem Double-click to make the submission file, then validate and score it if the organisers'
rem repository is next to this one.
rem
rem Match data: put the three BAR-ATH_tactical_* files in data\tactical\ inside this folder
rem (or in data\ directly). If they are not there, the folder that holds this one is tried.
rem The organisers' repository, if present, must be a folder named mtah next to this one.
setlocal
set "HERE=%~dp0"
set "PARENT=%HERE%.."
set /p TEAM=<"%HERE%team.txt"
set "PY=%HERE%.venv\Scripts\python.exe"

set "BASE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%BASE%" set "BASE=py"

if not exist "%PY%" (
  echo Making a Python environment for the hackathon ...
  "%BASE%" -m venv "%HERE%.venv" || goto fail
)

echo Installing pinned numpy and pandas (only slow the first time) ...
"%PY%" -m pip install --disable-pip-version-check -q -r "%HERE%requirements.txt" || goto fail

set "WHERE=--data-dir "%HERE%data""
if not exist "%HERE%data\tactical\BAR-ATH_tactical_atd.txt" if not exist "%HERE%data\BAR-ATH_tactical_atd.txt" set "WHERE=--tactical-dir "%PARENT%""

echo.
"%PY%" "%HERE%scripts\predict.py" %WHERE% || goto fail

if not exist "%PARENT%\mtah\evaluation\validate_submission.py" (
  echo.
  echo The organisers' repository was not found next to this folder, so the file was not
  echo validated or scored. Submission file: submission\%TEAM%.csv
  goto done
)

echo.
echo Checking the file the way the official scorer does ...
copy /y "%HERE%submission\%TEAM%.csv" "%PARENT%\mtah\evaluation\results\%TEAM%.csv" >nul
cd /d "%PARENT%\mtah"
"%PY%" evaluation\validate_submission.py "%HERE%submission\%TEAM%.csv" || goto fail

echo.
echo Scoring on the public 5 minutes ...
"%PY%" evaluation\run_evaluation.py || goto fail

:done
echo.
echo All steps finished. Submission file: submission\%TEAM%.csv
pause
exit /b 0

:fail
echo.
echo Something failed above. Take a screenshot of this window and send it to Claude.
pause
exit /b 1
